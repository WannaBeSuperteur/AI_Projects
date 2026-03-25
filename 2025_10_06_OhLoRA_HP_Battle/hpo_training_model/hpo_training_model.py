
import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
NUM_CLASSES = 10
EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA = 64
NUM_FEATURES_INPUT = 2 * EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA + NUM_CLASSES + 16
NUM_FEATURES_OUTPUT = 1


class HPOTrainingModel(nn.Module):
    def __init__(self):
        super(HPOTrainingModel, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(NUM_FEATURES_INPUT, 1024),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, NUM_FEATURES_OUTPUT),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_final(x)
        return x


class HPOTrainingDataset(Dataset):
    def __init__(self, dataset_df, dataset_name, tvt_type):
        self.dataset_df = dataset_df
        self.dataset_name = dataset_name
        self.tvt_type = tvt_type          # 'train', 'valid' or 'test'

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        return self.dataset_df.iloc[idx]
