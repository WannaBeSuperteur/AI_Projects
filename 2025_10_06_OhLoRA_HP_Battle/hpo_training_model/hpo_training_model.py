
import pandas as pd
import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
HPO_TRAINING_DATA_PATH = f'{PROJECT_DIR_PATH}/hpo_training_data/test'

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


# 학습 데이터셋을 merge 하여 최종 데이터셋 생성
# Create Date : 2026.03.29
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - merged_dataset_df (Pandas DataFrame) : 해당 dataset name 에 대한 HPO model 의 최종 데이터셋

def merge_dataset_df(dataset_name):
    csv_path = f'{HPO_TRAINING_DATA_PATH}/{dataset_name}'
    csv_names = os.listdir(csv_path)
    dfs = []

    for csv_name in csv_names:
        if 'hpo_model_train_dataset_df' in csv_name:
            df_path = os.path.join(csv_path, csv_name)
            df = pd.read_csv(df_path)
            dfs.append(df)

    merged_dataset_df = pd.concat(dfs, ignore_index=True)
    return merged_dataset_df


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        merged_dataset_df = merge_dataset_df(dataset_name)
        merged_dataset_size = len(merged_dataset_df)
        merged_dataset_train_size = int(0.9 * merged_dataset_size)

        train_df = merged_dataset_df.iloc[:merged_dataset_train_size, :]
        test_df = merged_dataset_df.iloc[merged_dataset_train_size:, :]

        train_dataset = HPOTrainingDataset(dataset_df=train_df, dataset_name=dataset_name, tvt_type='train')
        test_dataset = HPOTrainingDataset(dataset_df=train_df, dataset_name=dataset_name, tvt_type='test')

        print(train_df)
        print(test_df)
