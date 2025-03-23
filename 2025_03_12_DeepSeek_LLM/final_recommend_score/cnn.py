import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import random
import pandas as pd
import numpy as np

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


class BaseScoreCNN(nn.Module):
    def __init__(self):
        super(BaseScoreCNN, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 7 * 7, 256),
            nn.Sigmoid(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Conv
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.pool4(x)
        x = self.conv2(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(-1, 256 * 7 * 7)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# 데이터셋 로딩
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터 정보가 저장된 Pandas DataFrame
#                                   columns = ['img_path', 'score']

# Returns:
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader

def load_dataset(dataset_df):
    raise NotImplementedError


# 모델 학습 실시 (Stratified K-Fold)
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader

# Returns:
# - cnn_models (list(nn.Module)) : 학습된 CNN Model 의 리스트 (총 K 개의 모델)

def train_cnn(data_loader):
    raise NotImplementedError


# 모델 불러오기
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - cnn_models (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

def load_cnn_model():
    raise NotImplementedError


# 학습된 모델을 이용하여 기본 가독성 점수 예측 (Ensemble 의 아이디어 / K 개 모델의 평균으로)
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - img_paths  (list(str))       : 기본 가독성 점수 예측 대상 이미지의 경로 리스트
# - cnn_models (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

# Returns:
# - final_score (Pandas DataFrame) : 최종 예측 점수 (0.0 ~ 5.0) 를 저장한 Pandas DataFrame
#                                    columns = ['img_path', 'score']

def predict_score(img_paths, cnn_models):
    raise NotImplementedError


if __name__ == '__main__':

    # load dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/final_recommend_score/scores.csv')
    data_loader = load_dataset(dataset_df)

    # load or train model
    try:
        print('loading CNN models ...')
        cnn_models = load_cnn_model()

    except Exception as e:
        print(f'CNN model load failed : {e}')
        cnn_models = train_cnn(data_loader)

    # predict using CNN model
    base_dir = f'{PROJECT_DIR_PATH}/final_recommend_score/training_data/base'
    sft_generated_dir = f'{PROJECT_DIR_PATH}/final_recommend_score/training_data/sft_generated_orpo_dataset'

    img_paths_base = [f'{base_dir}/diagram_{i:%06d}.png' for i in range(5)]
    img_paths_sft_generated = [f'{sft_generated_dir}/diagram_{i:%06d}.png' for i in range(50)]
    img_paths = img_paths_base + img_paths_sft_generated

    final_score = predict_score(img_paths, cnn_models)

    # save and print prediction result
    os.makedirs(f'{PROJECT_DIR_PATH}/final_recommend_score/log', exist_ok=True)
    final_score.to_csv('log/final_score.csv')

    print('FINAL PREDICTION :\n')
    print(final_score)
