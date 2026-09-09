
import os

import pandas as pd

import torch
from torch.utils.data import Dataset, random_split
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# to prevent force system off during S-BERT training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

SBERT_TRAIN_BATCH_SIZE = 16
SBERT_VALID_BATCH_SIZE = 4
SBERT_EPOCHS = 7

SBERT_MODEL_SAVE_PATH = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/trained_sbert_model'
SBERT_MODEL_CKPT_PATH = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/checkpoints'


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
            "prob": torch.tensor(self.probs[idx], dtype=torch.long)
        }


class EmbeddingProbPredictor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.predictor = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pooling(outputs, attention_mask)
        prob = self.predictor(emb)
        return prob


class EmbeddingProbTrainer(nn.Module):
    def __init__(self, predictor, datasets: dict):
        super().__init__()
        self.predictor = predictor
        self.classifier = nn.Linear(predictor.config.hidden_size, 1)
        self.datasets = datasets


def train_probability_predictor(model_path: str, dataset_path: str):
    """train text embedding probability predictor."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    predictor = EmbeddingProbPredictor(model)
    dataset_df = pd.read_csv(dataset_path)
    dataset_size = len(dataset_df)
    dataset = SingleTextDataset(dataset_df, tokenizer)

    n_train_size = int(0.75 * dataset_size)
    n_valid_size = int(0.125 * dataset_size)
    n_test_size = dataset_size - (n_train_size + n_valid_size)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [n_train_size, n_valid_size, n_test_size])
    datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}

    trainer = EmbeddingProbTrainer(predictor, datasets)


if __name__ == '__main__':
    model_path = "codefuse-ai/F2LLM-v2-330M"
    dataset_path = os.path.join(PROJECT_DIR_PATH,
                                "code_reviewer",
                                "ai_dataset",
                                "dataset_01_func_docstring_single_responsibility.csv")

    train_probability_predictor(model_path, dataset_path)
