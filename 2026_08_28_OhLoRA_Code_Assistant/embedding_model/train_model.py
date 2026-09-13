import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import sklearn

import math
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import random_split, DataLoader

import datasets
from sentence_transformers import SentenceTransformer, util, SentenceTransformerTrainingArguments, \
                                  SentenceTransformerTrainer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from transformers import AutoTokenizer, AutoModel, EarlyStoppingCallback, TrainerCallback

from sklearn.metrics.pairwise import cosine_similarity


np.set_printoptions(linewidth=160)
torch.manual_seed(2026)

# to prevent force system off during S-BERT training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

MAX_EPOCHS_PROB = 20
EARLY_STOPPING_PATIENCE_PROB = 5

MAX_EPOCHS_SIMILARITY = 12
EARLY_STOPPING_PATIENCE_SIMILARITY = 3

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


class SingleTextDataset(torch.utils.data.Dataset):
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
        val_mse_sum, val_mae_sum, val_loss_sum = 0.0, 0.0, 0.0

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
                val_mae_batch = sklearn.metrics.mean_absolute_error(preds, prob_labels)
                val_mae_sum += val_mae_batch

                total += prob_labels.shape[0]

        val_mse = val_mse_sum / total
        val_mae = val_mae_sum / total
        val_loss = val_loss_sum / total

        return val_mse, val_mae, val_loss

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
            'valid_mae': [],
            'valid_loss': [],
            'torch_memory': []
        }

        while True:
            start_at = time.time()

            self._run_train()
            valid_mse, valid_mae, valid_loss = self._run_validation_or_test(model=self.predictor,
                                                                            data_loader=self.valid_loader)

            print(f'epoch={self.current_epoch}, ' +
                  f'val_mse={valid_mse:.6f}, val_mae={valid_mae:.6f}, val_loss={valid_loss:.6f}')
            val_loss_list.append(valid_loss)

            if self.predictor.scheduler is not None:
                self.predictor.scheduler.step()

            # update best epoch model
            if min_valid_loss is None or valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                min_valid_loss_epoch = self.current_epoch

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
            train_log['valid_mae'].append(round(valid_mae, 6))
            train_log['valid_loss'].append(round(valid_loss, 6))
            train_log['torch_memory'].append(torch.cuda.memory_allocated())
            pd.DataFrame(train_log).to_csv(train_log_path)

            if (self.current_epoch + 1 >= MAX_EPOCHS_PROB or
                self.current_epoch - min_valid_loss_epoch >= EARLY_STOPPING_PATIENCE_PROB):
                break

            self.current_epoch += 1

        # run test
        print('testing ...')

        test_start_at = time.time()
        test_mse, test_mae, test_loss = self._run_validation_or_test(model=best_epoch_model,
                                                                     data_loader=self.test_loader)

        train_log['epoch'].append('test')
        train_log['epoch_time'].append(round(time.time() - test_start_at, 3))
        train_log['valid_mse'].append(test_mse)
        train_log['valid_mae'].append(test_mae)
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
    dataset_df = dataset_df.sample(frac=1)
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


def create_datasets_for_tvt(dataset_df: pd.DataFrame):
    dataset_size = len(dataset_df)
    n_train_size = int(0.75 * dataset_size)
    n_valid_size = int(0.125 * dataset_size)

    formatted_df = dataset_df.rename(columns={
        'code_1': 'sentence1',
        'code_2': 'sentence2',
        'similarity': 'label'
    })[['sentence1', 'sentence2', 'label']]

    train_df = formatted_df[:n_train_size]
    valid_df = formatted_df[n_train_size:n_train_size + n_valid_size]
    test_df = formatted_df[n_train_size + n_valid_size:]

    train_dataset = datasets.Dataset.from_pandas(train_df, preserve_index=False)
    valid_dataset = datasets.Dataset.from_pandas(valid_df, preserve_index=False)
    test_dataset = datasets.Dataset.from_pandas(test_df, preserve_index=False)

    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }


def test_similarity_predictor(model_dir_path: str, device: str, test_dataset):
    """test text embedding probability predictor."""

    best_model = SentenceTransformer(model_dir_path, device=device, trust_remote_code=True)
    best_model.eval()

    with torch.no_grad():
        predicted_scores, true_labels = valid_or_test_similarity_predictor(model=best_model,
                                                                           val_or_test_dataset=test_dataset)

    test_mse = sklearn.metrics.mean_squared_error(predicted_scores, true_labels)
    test_mae = sklearn.metrics.mean_absolute_error(predicted_scores, true_labels)
    return test_mse, test_mae


def valid_or_test_similarity_predictor(model, val_or_test_dataset):
    predicted_scores, true_labels = [], []

    for batch in val_or_test_dataset:
        sentence1, sentence2, label = batch['sentence1'], batch['sentence2'], batch['label']

        emb1 = np.array([model.encode(sentence1)])
        emb2 = np.array([model.encode(sentence2)])
        similarity = cosine_similarity(emb1, emb2)

        predicted_scores.extend(similarity[0].tolist())
        true_labels.append(label)

    return predicted_scores, true_labels


class LogTrainingCallback(TrainerCallback):
    def __init__(self, log_function, metric_key="eval_spearman_cosine"):
        self.log_function = log_function
        self.metric_key = metric_key

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            score = metrics.get(self.metric_key, metrics.get("eval_valid_spearman_cosine", 0.0))
            epoch = state.epoch if state.epoch is not None else 0.0
            steps = state.global_step
            self.log_function(score, epoch, steps)


def train_similarity_predictor(model_path: str, dataset_path: str, task_name: str):
    """train text embedding similarity predictor."""

    train_log = {
        'epochs': [],
        'steps': [],
        'valid_similarity_score': [],
        'valid_mse': [],
        'valid_mae': []
    }

    def log_training(score: float, epoch: float, steps: int):
        with torch.no_grad():
            predicted_scores, true_labels = valid_or_test_similarity_predictor(model=model,
                                                                               val_or_test_dataset=valid_dataset)

        valid_mse = sklearn.metrics.mean_squared_error(predicted_scores, true_labels)
        valid_mae = sklearn.metrics.mean_absolute_error(predicted_scores, true_labels)

        train_log['epochs'].append(round(epoch, 2))
        train_log['steps'].append(steps)
        train_log['valid_similarity_score'].append(round(score, 6))
        train_log['valid_mse'].append(round(valid_mse, 6))
        train_log['valid_mae'].append(round(valid_mae, 6))
        pd.DataFrame(train_log).to_csv(train_log_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    train_log_path = os.path.join(TRAIN_LOG_PATH, f'{task_name}.csv')

    dataset_df = pd.read_csv(dataset_path)
    dataset_df = dataset_df.sample(frac=1)
    datasets = create_datasets_for_tvt(dataset_df)
    train_dataset, valid_dataset, test_dataset = datasets['train'], datasets['valid'], datasets['test']

    valid_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=valid_dataset['sentence1'],
        sentences2=valid_dataset['sentence2'],
        scores=valid_dataset['label'],
        name='valid'
    )

    train_loss = losses.CoSENTLoss(model=model)

    model_dir_path = os.path.join(MODEL_SAVE_PATH, task_name)
    os.makedirs(model_dir_path, exist_ok=True)

    steps_per_epoch = math.ceil(len(train_dataset) / 2)
    total_train_steps = steps_per_epoch * 5

    warmup_fraction = LEARNING_RATE[model_path]['warmup_fraction']
    warmup_steps = int(total_train_steps * warmup_fraction)
    base_lr = LEARNING_RATE[model_path]['lr']

    training_args = SentenceTransformerTrainingArguments(
        output_dir=model_dir_path,
        num_train_epochs=MAX_EPOCHS_SIMILARITY,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=125,
        learning_rate=base_lr,
        warmup_steps=warmup_steps,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        save_strategy="no"
    )

    early_stopping_patience = steps_per_epoch * EARLY_STOPPING_PATIENCE_SIMILARITY
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['valid'],
        loss=train_loss,
        evaluator=valid_evaluator,
        callbacks=[LogTrainingCallback(log_training),
                   EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    trainer.train()
    trainer.save_model(model_dir_path)

    test_mse, test_mae = test_similarity_predictor(model_dir_path, device, test_dataset)

    train_log['epochs'].append('test')
    train_log['steps'].append('test')
    train_log['valid_similarity_score'].append('')
    train_log['valid_mse'].append(round(test_mse, 6))
    train_log['valid_mae'].append(round(test_mae, 6))
    pd.DataFrame(train_log).to_csv(train_log_path)


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="task name (e.g. 01_unnecessary_prints)")
    args = parser.parse_args()

    task_name = args.task
    task_info = task_name_to_info[task_name]

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
