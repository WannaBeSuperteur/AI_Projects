
import os

import numpy as np
import pandas as pd
import sklearn

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


np.set_printoptions(linewidth=160)
torch.manual_seed(2026)

# to prevent force system off during S-BERT training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
EPOCHS = 7

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

MODEL_SAVE_PATH = f'{PROJECT_DIR_PATH}/embedding_model/models'
MODEL_CKPT_PATH = f'{PROJECT_DIR_PATH}/embedding_model/checkpoints'

HIDDEN_SIZE = {"codefuse-ai/F2LLM-v2-330M": 896}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SingleTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = list(df['code'])
        self.probs = list(df['probability'])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "prob": torch.tensor(self.probs[idx], dtype=torch.float16)
        }


class EmbeddingProbPredictor(nn.Module):
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.final_linear = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pooling(outputs, attention_mask)
        prob = self.final_linear(emb)
        return prob


class EmbeddingProbTrainer:
    def __init__(self, predictor: EmbeddingProbPredictor, data_loaders: dict, task_name: str):
        super().__init__()
        self.task_name = task_name

        self.predictor = predictor
        self.predictor.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=5e-5)
        self.predictor.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.predictor.optimizer,
                                                                          gamma=0.95)
        self.loss_func = nn.BCEWithLogitsLoss()

        self.data_loaders = data_loaders
        self.train_loader = self.data_loaders['train']
        self.valid_loader = self.data_loaders['valid']
        self.test_loader = self.data_loaders['test']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor.to(self.device)

    def _run_train(self):
        self.predictor.train()

        total = 0
        train_loss_sum = 0.0

        for idx, items in enumerate(self.train_loader):
            items = {k: v.to(self.device) for k, v in items.items()}
            inputs, attention_mask, prob_labels = items['input_ids'], items['attention_mask'], items['prob']
            prob_labels = prob_labels.reshape(-1, 1)

            # train 실시
            self.predictor.optimizer.zero_grad()
            outputs = self.predictor(inputs, attention_mask).to(torch.float32)

            loss = self.loss_func(outputs, prob_labels)
            loss.backward()
            self.predictor.optimizer.step()

            prob_labels = prob_labels.detach().cpu().numpy()

            train_loss_sum += loss.item()
            total += prob_labels.shape[0]

        train_loss = train_loss_sum / total
        return train_loss

    def _run_validation_or_test(self, model: nn.Module, data_loader: DataLoader):
        model.eval()
        total = 0
        val_mse_sum, val_loss_sum = 0.0, 0.0

        with torch.no_grad():
            for idx, items in enumerate(data_loader):
                items = {k: v.to(self.device) for k, v in items.items()}

                inputs, attention_mask, prob_labels = items['input_ids'], items['attention_mask'], items['prob']
                outputs = self.predictor(inputs, attention_mask).to(torch.float32)
                preds = torch.sigmoid(outputs)
                prob_labels = prob_labels.reshape(-1, 1)

                val_loss_batch = self.loss_func(outputs, prob_labels)
                val_loss_sum += val_loss_batch

                preds = preds.detach().cpu()
                prob_labels = prob_labels.detach().cpu()

                val_mse_batch = sklearn.metrics.mean_squared_error(preds, prob_labels)
                val_mse_sum += val_mse_batch

                total += prob_labels.shape[0]

        val_mse = val_mse_sum / total
        val_loss = val_loss_sum / total

        return val_mse, val_loss

    def _run_all_process(self):
        self.current_epoch = 0
        min_valid_loss_epoch = -1  # Loss-based Early Stopping
        min_valid_loss = None
        best_epoch_model = None
        val_loss_list = []

        while True:
            self._run_train()
            valid_mse, valid_loss = self._run_validation_or_test(model=self.predictor,
                                                                 data_loader=self.valid_loader)

            print(f'epoch={self.current_epoch}, val_mse={valid_mse:.6f}, val_loss={valid_loss:.6f}')
            val_loss_list.append(valid_loss)

            if self.predictor.scheduler is not None:
                self.predictor.scheduler.step()

            # update best epoch model
            if min_valid_loss is None or valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                min_valid_loss_epoch = self.current_epoch
                best_epoch_model_valid_mse = valid_mse

                pretrained_model = EmbeddingProbPredictor(base_model=self.predictor.base_model,
                                                          hidden_size=self.predictor.hidden_size)

                best_epoch_model = pretrained_model.to(self.device)
                best_epoch_model.device = self.device
                best_epoch_model.load_state_dict(self.predictor.state_dict())

                ckpt_dir_path = os.path.join(MODEL_CKPT_PATH, self.task_name)
                os.makedirs(ckpt_dir_path, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir_path, f"epoch_{self.current_epoch:04d}.pth")
                torch.save(best_epoch_model.state_dict(), ckpt_path)

            if self.current_epoch + 1 >= MAX_EPOCHS or self.current_epoch - min_valid_loss_epoch >= EARLY_STOPPING_PATIENCE:
                break

            self.current_epoch += 1

        # assert best epoch model accuracy & loss
        checked_valid_mse, checked_valid_loss = self._run_validation_or_test(model=best_epoch_model,
                                                                             data_loader=self.valid_loader)

        print(f'[best model] val_mse={best_epoch_model_valid_mse}, val_loss={min_valid_loss}')
        print(f'[check] val_mse={checked_valid_mse}, val_loss={checked_valid_loss}')

        assert abs(best_epoch_model_valid_mse - checked_valid_mse) <= 1e-6
        assert abs(min_valid_loss - checked_valid_loss) <= 1e-6

        # run test
        print('testing ...')

        test_mse, test_loss = self._run_validation_or_test(model=best_epoch_model,
                                                           data_loader=self.test_loader)

        model_dir_path = os.path.join(MODEL_SAVE_PATH, self.task_name)
        os.makedirs(model_dir_path, exist_ok=True)
        model_path = os.path.join(model_dir_path, f"epoch_{self.current_epoch:%04d}.pth")
        torch.save(best_epoch_model.state_dict(), model_path)

        return val_loss_list, test_mse, best_epoch_model

    def run(self):
        self._run_all_process()


def train_probability_predictor(model_path: str, dataset_path: str, task_name: str):
    """train text embedding probability predictor."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    hidden_size = HIDDEN_SIZE[model_path]

    predictor = EmbeddingProbPredictor(model, hidden_size)
    dataset_df = pd.read_csv(dataset_path)
    dataset_size = len(dataset_df)
    dataset = SingleTextDataset(dataset_df, tokenizer)

    n_train_size = int(0.75 * dataset_size)
    n_valid_size = int(0.125 * dataset_size)
    n_test_size = dataset_size - (n_train_size + n_valid_size)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [n_train_size, n_valid_size, n_test_size])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    data_loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    trainer = EmbeddingProbTrainer(predictor, data_loaders, task_name)
    trainer.run()


def create_samples_and_dataloaders_for_tvt(dataset_df: pd.DataFrame):
    dataset_size = len(dataset_df)
    n_train_size = int(0.75 * dataset_size)
    n_valid_size = int(0.125 * dataset_size)

    def create_samples_and_dataloader(df: pd.DataFrame, shuffle: bool):
        samples = [
            InputExample(texts=[row['code_1'], row['code_2']], label=row['similarity'])
            for _, row in df.iterrows()
        ]
        return samples, DataLoader(samples, shuffle=shuffle, batch_size=2)

    train_df = dataset_df[:n_train_size]
    valid_df = dataset_df[n_train_size:n_train_size + n_valid_size]
    test_df = dataset_df[n_train_size + n_valid_size:]

    _, train_dataloader = create_samples_and_dataloader(train_df, shuffle=True)
    valid_samples, _ = create_samples_and_dataloader(valid_df, shuffle=False)
    _, test_dataloader = create_samples_and_dataloader(test_df, shuffle=False)

    return {
        'train_loader': train_dataloader,
        'test_loader': test_dataloader,
        'valid_samples': valid_samples
    }


def test_similarity_predictor(model_dir_path: str, device: str, test_dataloader: DataLoader):
    """test text embedding probability predictor."""

    best_model = SentenceTransformer(model_dir_path, device=device)
    best_model.eval()

    true_labels = []
    predicted_scores = []

    with torch.no_grad():
        for batch in test_dataloader:
            texts1 = [ex.texts[0] for ex in batch]
            texts2 = [ex.texts[1] for ex in batch]
            labels = [ex.label for ex in batch]

            embeddings1 = best_model.encode(texts1, convert_to_tensor=True, show_progress_bar=False)
            embeddings2 = best_model.encode(texts2, convert_to_tensor=True, show_progress_bar=False)
            cos_sim = best_model.similarity(embeddings1, embeddings2)
            preds = torch.diagonal(cos_sim).cpu().numpy()

            predicted_scores.extend(preds)
            true_labels.extend(labels)

            print(f'preds : {preds}')
            print(f'labels : {true_labels}')

    mse_result = sklearn.metrics.mean_squared_error(true_labels, predicted_scores)
    return mse_result


def train_similarity_predictor(model_path: str, dataset_path: str, task_name: str):
    """train text embedding similarity predictor."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)

    dataset_df = pd.read_csv(dataset_path)
    dataset_df = dataset_df.sample(frac=1)
    samples_and_dataloaders = create_samples_and_dataloaders_for_tvt(dataset_df)

    valid_samples = samples_and_dataloaders['valid_samples']
    valid_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=[s.texts[0] for s in valid_samples],
        sentences2=[s.texts[1] for s in valid_samples],
        scores=[s.label for s in valid_samples],
        name="valid-eval"
    )

    train_loss = losses.CosineSimilarityLoss(model=model)

    model_dir_path = os.path.join(MODEL_SAVE_PATH, task_name)
    os.makedirs(model_dir_path, exist_ok=True)

    train_dataloader = samples_and_dataloaders['train_loader']
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=valid_evaluator,
        epochs=10,
        evaluation_steps=10,
        warmup_steps=5,
        output_path=model_dir_path
    )

    test_dataloader = samples_and_dataloaders['test_loader']
    test_similarity_predictor(model_dir_path, device, test_dataloader)


if __name__ == '__main__':
    model_path = "codefuse-ai/F2LLM-v2-330M"

#    dataset_path = os.path.join(PROJECT_DIR_PATH,
#                                "code_reviewer",
#                                "ai_dataset",
#                                "dataset_01_func_docstring_single_responsibility.csv")
#    task_name = "func_docstring_single_responsibility"
#    train_probability_predictor(model_path, dataset_path, task_name)

    dataset_path = os.path.join(PROJECT_DIR_PATH,
                                "code_reviewer",
                                "ai_dataset",
                                "dataset_01_return_matched_with_func_name.csv")
    task_name = "return_matched_with_func_name"
    train_similarity_predictor(model_path, dataset_path, task_name)
