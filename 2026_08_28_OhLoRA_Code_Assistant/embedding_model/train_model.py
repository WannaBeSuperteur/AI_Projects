
import os
import shutil
import time

import numpy as np
import pandas as pd
import sklearn

import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util import cos_sim
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainerCallback

np.set_printoptions(linewidth=160)
torch.manual_seed(2026)

# to prevent force system off during S-BERT training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
EPOCHS = 7

MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

MODEL_SAVE_PATH = f'{PROJECT_DIR_PATH}/embedding_model/models'
MODEL_CKPT_PATH = f'{PROJECT_DIR_PATH}/embedding_model/checkpoints'
TRAIN_LOG_PATH = f'{PROJECT_DIR_PATH}/embedding_model/train_log'

GTE_MODERNBERT_BASE = 'Alibaba-NLP/gte-modernbert-base'
GIGA_EMBEDDINGS_INSTRUCT = 'ai-sage/Giga-Embeddings-instruct-480M-0826'
F2LLM_V2_330M = 'codefuse-ai/F2LLM-v2-330M'

HIDDEN_SIZE = {GTE_MODERNBERT_BASE: 768,
               GIGA_EMBEDDINGS_INSTRUCT: 1024,
               F2LLM_V2_330M: 896}

LEARNING_RATE = {GTE_MODERNBERT_BASE: {'lr': 3e-5, 'warmup_fraction': 0.075},
                 GIGA_EMBEDDINGS_INSTRUCT: {'lr': 3e-5, 'warmup_fraction': 0.01},
                 F2LLM_V2_330M: {'lr': 2.5e-6, 'warmup_fraction': 0.4}}


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
                val_loss_sum += float(val_loss_batch.detach().cpu())

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

        ckpt_dir_path = os.path.join(MODEL_CKPT_PATH, self.task_name)
        model_dir_path = os.path.join(MODEL_SAVE_PATH, self.task_name)
        train_log_path = os.path.join(TRAIN_LOG_PATH, f'{self.task_name}.csv')

        train_log = {
            'epoch': [],
            'epoch_time': [],
            'valid_mse': [],
            'valid_loss': [],
            'torch_memory': []
        }

        while True:
            start_at = time.time()

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

                if os.path.exists(ckpt_dir_path):
                    shutil.rmtree(ckpt_dir_path)

                os.makedirs(ckpt_dir_path, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir_path, f"epoch_{self.current_epoch:04d}.pth")
                torch.save(best_epoch_model.state_dict(), ckpt_path)

            train_log['epoch'].append(self.current_epoch)
            train_log['epoch_time'].append(round(time.time() - start_at, 3))
            train_log['valid_mse'].append(round(valid_mse, 6))
            train_log['valid_loss'].append(round(valid_loss, 6))
            train_log['torch_memory'].append(torch.cuda.memory_allocated())
            pd.DataFrame(train_log).to_csv(train_log_path)

            if self.current_epoch + 1 >= MAX_EPOCHS or self.current_epoch - min_valid_loss_epoch >= EARLY_STOPPING_PATIENCE:
                break

            self.current_epoch += 1

        # run test
        print('testing ...')

        test_start_at = time.time()
        test_mse, test_loss = self._run_validation_or_test(model=best_epoch_model,
                                                           data_loader=self.test_loader)

        train_log['epoch'].append('test')
        train_log['epoch_time'].append(round(time.time() - test_start_at, 3))
        train_log['valid_mse'].append(test_mse)
        train_log['valid_loss'].append(test_loss)
        train_log['torch_memory'].append(torch.cuda.memory_allocated())
        pd.DataFrame(train_log).to_csv(train_log_path)

        if os.path.exists(ckpt_dir_path):
            shutil.rmtree(ckpt_dir_path)

        os.makedirs(model_dir_path, exist_ok=True)
        model_path = os.path.join(model_dir_path, f"epoch_{self.current_epoch:04d}.pth")
        torch.save(best_epoch_model.state_dict(), model_path)

    def run(self):
        self._run_all_process()


def train_probability_predictor(model_path: str, dataset_path: str, task_name: str):
    """train text embedding probability predictor."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
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

    best_model = SentenceTransformer(model_dir_path, device=device, trust_remote_code=True)
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
            cos_sims = best_model.similarity(embeddings1, embeddings2)
            preds = torch.diagonal(cos_sims).cpu().numpy()

            predicted_scores.extend(preds)
            true_labels.extend(labels)

            print(f'preds : {preds}')
            print(f'labels : {true_labels}')

    test_mse = sklearn.metrics.mean_squared_error(true_labels, predicted_scores)
    return test_mse


def train_similarity_predictor(model_path: str, dataset_path: str, task_name: str):
    """train text embedding similarity predictor."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    test_log_path = os.path.join(TRAIN_LOG_PATH, f'{task_name}.csv')

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

    train_loss = losses.CoSENTLoss(model=model)

    model_dir_path = os.path.join(MODEL_SAVE_PATH, task_name)
    os.makedirs(model_dir_path, exist_ok=True)

    train_dataloader = samples_and_dataloaders['train_loader']
    total_train_steps = len(train_dataloader) * 5
    warmup_fraction = LEARNING_RATE[model_path]['warmup_fraction']
    warmup_steps = int(total_train_steps * warmup_fraction)
    base_lr = LEARNING_RATE[model_path]['lr']

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=valid_evaluator,
        epochs=5,
        evaluation_steps=50,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": base_lr},
        output_path=model_dir_path
    )

    test_dataloader = samples_and_dataloaders['test_loader']

    test_start_at = time.time()
    test_mse = test_similarity_predictor(model_dir_path, device, test_dataloader)
    test_log = {
        'test_time': [round(time.time() - test_start_at, 3)],
        'test_mse': [test_mse]
    }
    pd.DataFrame(test_log).to_csv(test_log_path)


if __name__ == '__main__':
    os.makedirs(TRAIN_LOG_PATH, exist_ok=True)

    task_name_to_info = {
        '01_unnecessary_prints': {'model_path': F2LLM_V2_330M, 'task_type': 'prob'},
        '01_similar_variables': {'model_path': GIGA_EMBEDDINGS_INSTRUCT, 'task_type': 'sim'},
        '01_names': {'model_path': GIGA_EMBEDDINGS_INSTRUCT, 'task_type': 'prob'},
        '01_return_matched_with_func_name': {'model_path': GIGA_EMBEDDINGS_INSTRUCT, 'task_type': 'sim'},
        '01_func_docstring_single_responsibility': {'model_path': F2LLM_V2_330M, 'task_type': 'prob'},
        '01_func_docstring_docstring_and_name': {'model_path': F2LLM_V2_330M, 'task_type': 'sim'},
        '04_func_args_bindable': {'model_path': GIGA_EMBEDDINGS_INSTRUCT, 'task_type': 'prob'},
        '04_func_args_dynamic': {'model_path': GIGA_EMBEDDINGS_INSTRUCT, 'task_type': 'prob'},
        '06_refactor_into_class_case_2_state_vars_if_else': {'model_path': GIGA_EMBEDDINGS_INSTRUCT, 'task_type': 'prob'},
        '06_similar_function_names': {'model_path': GIGA_EMBEDDINGS_INSTRUCT, 'task_type': 'sim'}
    }

    for task_name, task_info in task_name_to_info.items():
        dataset_path = os.path.join(PROJECT_DIR_PATH,
                                    "code_reviewer",
                                    "ai_dataset",
                                    f"dataset_{task_name}.csv")

        model_path = task_info['model_path']
        task_type = task_info['task_type']

        if task_type == 'prob':
            train_probability_predictor(model_path, dataset_path, task_name)
        elif task_type == 'sim':
            train_similarity_predictor(model_path, dataset_path, task_name)
