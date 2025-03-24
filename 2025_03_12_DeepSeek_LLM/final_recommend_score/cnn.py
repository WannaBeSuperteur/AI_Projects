import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import save_image
from torchinfo import summary

from sklearn.model_selection import KFold
import pandas as pd
import cv2
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from global_common.torch_training import run_train, run_validation


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/training_data'
K_FOLDS = 5
EARLY_STOPPING_ROUNDS = 10
IMG_HEIGHT = 128
IMG_WIDTH = 128


class BaseScoreCNN(nn.Module):
    def __init__(self):
        super(BaseScoreCNN, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv1_center = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x[:, :, IMG_HEIGHT // 4 : 3 * IMG_HEIGHT // 4, IMG_WIDTH // 4 : 3 * IMG_WIDTH // 4]

        # Conv
        x = self.conv1(x)  # 62
        x = self.conv2(x)  # 60
        x = self.pool1(x)  # 30

        x = self.conv3(x)  # 28
        x = self.pool2(x)  # 14

        x = self.conv4(x)  # 12
        x = self.pool3(x)  # 6

        x = self.conv5(x)  # 4

        x = x.view(-1, 256 * 4 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


class DiagramImageDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_paths = dataset_df['img_path'].tolist()
        self.scores = dataset_df['score'].tolist()
        self.transform = transform

    def __len__(self):
        return 8 * len(self.img_paths)

    def __getitem__(self, idx):
        true_idx = idx // (4 * 2)
        rotate_angle = idx % 4        # 0 (원본), 1 (반시계 90도), 2 (180도), 3 (시계 90도) 회전
        flip_option = (idx // 4) % 2  # 0 (no flip), 1 (vertical flip)

        img_path = f'{TRAIN_DATA_DIR_PATH}/{self.img_paths[true_idx]}'
        image = read_image(img_path)

        # resize and normalize
        image = self.transform(image)
        score = self.scores[true_idx] / 5.0

        # rotate
        if rotate_angle == 1:
            image = v2.functional.rotate(image, 90)

        elif rotate_angle == 2:
            image = v2.functional.rotate(image, 180)

        elif rotate_angle == 3:
            image = v2.functional.rotate(image, 270)

        # flip
        if flip_option == 1:
            image = v2.functional.vertical_flip(image)

        return image, score


# 데이터셋 로딩
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터 정보가 저장된 Pandas DataFrame
#                                   columns = ['img_path', 'score']

# Returns:
# - train_loader (DataLoader) : Train 데이터셋을 로딩한 PyTorch DataLoader
# - train_loader (DataLoader) : Test 데이터셋을 로딩한 PyTorch DataLoader

def load_dataset(dataset_df):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    diagram_image_dataset = DiagramImageDataset(dataset_df, transform=transform)

    dataset_size = len(diagram_image_dataset)
    train_size = int(0.8 * dataset_size)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, dataset_size))

    train_dataset = Subset(diagram_image_dataset, train_indices)
    test_dataset = Subset(diagram_image_dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    print(f'size of train loader : {len(train_loader.dataset)}')
    print(f'size of train loader : {len(test_loader.dataset)}')

    # test code
    tvt_labels = ['train', 'test']
    loaders = [train_loader, test_loader]
    test_dir_path = f'{PROJECT_DIR_PATH}/final_recommend_score/temp_test'

    for tvt_label, loader in zip(tvt_labels, loaders):
        for i in range(40):
            img = loader.dataset.__getitem__(i)[0]
            os.makedirs(test_dir_path, exist_ok=True)
            save_image(img, f'{test_dir_path}/{tvt_label}_data_{i:02d}.png')

    return train_loader, test_loader


# 모델 학습 실시 (Stratified K-Fold)
# Create Date : 2025.03.23
# Last Update Date : 2025.03.24
# - 개행 수정

# Arguments:
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader

# Returns:
# - cnn_models (list(nn.Module)) : 학습된 CNN Model 의 리스트 (총 K 개의 모델)

def train_cnn(data_loader):

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training : {device}')

    if 'cuda' in str(device):
        print(f'cuda is available with device {torch.cuda.get_device_name()}')

    # create models
    cnn_models = []

    for i in range(K_FOLDS):
        cnn_model = BaseScoreCNN()
        cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.001)
        cnn_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=cnn_model.optimizer,
                                                                         T_max=10,
                                                                         eta_min=0)
        cnn_models.append(cnn_model)

    summary(cnn_models[0], input_size=(16, 3, IMG_HEIGHT, IMG_WIDTH))

    # Split train data using Stratified K-fold (5 folds)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)
    kfold_splitted_dataset = kfold.split(data_loader.dataset)

    # loss function
    loss_func = nn.BCELoss(reduction='sum')

    # run training and validation
    for fold, (train_idxs, valid_idxs) in enumerate(kfold_splitted_dataset):
        print(f'=== training model {fold + 1} / {K_FOLDS} ===')

        model = cnn_models[fold]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idxs)

        train_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=16, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=4, sampler=valid_sampler)

        current_epoch = 0
        min_loss_epoch = -1  # Loss-based Early Stopping
        min_loss = None

        while True:
            train_loss = run_train(model=model,
                                   train_loader=train_loader,
                                   device=device,
                                   loss_func=loss_func)

            _, val_loss = run_validation(model=model,
                                         valid_loader=valid_loader,
                                         device=device,
                                         loss_func=loss_func)

            print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}, val_loss : {val_loss:.4f}')

            model.scheduler.step()

            if min_loss is None or val_loss < min_loss - 0.001:
                min_loss = val_loss
                min_loss_epoch = current_epoch

            if current_epoch - min_loss_epoch >= EARLY_STOPPING_ROUNDS:
                break

            current_epoch += 1

        # save model
        model_save_path = f'{PROJECT_DIR_PATH}/final_recommend_score/models/model_{fold}'
        torch.save(model.state_dict(), model_save_path)

    return cnn_models


# 모델 불러오기
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - cnn_models (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

def load_cnn_model():
    raise NotImplementedError


# 학습된 모델을 이용하여 주어진 path 의 이미지에 대해 기본 가독성 점수 예측 (Ensemble 의 아이디어 / K 개 모델의 평균으로)
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - img_paths  (list(str))       : 기본 가독성 점수 예측 대상 이미지의 경로 리스트
# - cnn_models (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

# Returns:
# - final_score (Pandas DataFrame) : 최종 예측 점수 및 각 모델별 예측 점수 (0.0 ~ 5.0) 를 저장한 Pandas DataFrame
#                                    columns = ['img_path', 'score', 'model0_score', 'model1_score', ...]

def predict_score_path(img_paths, cnn_models):
    raise NotImplementedError


# 학습된 모델을 이용하여 주어진 이미지에 대해 기본 가독성 점수 예측 (Ensemble 의 아이디어 / K 개 모델의 평균으로)
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - test_loader (DataLoader)      : 테스트 데이터셋을 로딩한 PyTorch DataLoader
# - cnn_models  (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

# Returns:
# - final_score        (Pandas DataFrame) : 최종 예측 점수 및 각 모델별 예측 점수 (0.0 ~ 5.0) 를 저장한 Pandas DataFrame
#                                           columns = ['img_path', 'score', 'model0_score', 'model1_score', ...]
# - performance_scores (dict)             : 모델 성능 평가 결과 (0 ~ 5 점 척도 기준의 MAE, MSE, RMSE Error)
#                                           {'mae': float, 'mse': float, 'rmse': float}

def predict_score_image(test_loader, cnn_models):
    raise NotImplementedError


if __name__ == '__main__':

    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/final_recommend_score/scores.csv')

    # resize images to (128, 128)
    img_paths = dataset_df['img_path'].tolist()

    for idx, img_path in enumerate(img_paths):
        if idx % 100 == 0:
            print(f'resizing diagram image progress : {idx}')

        img_full_path = f'{TRAIN_DATA_DIR_PATH}/{img_path}'
        img = cv2.imread(img_full_path, cv2.IMREAD_COLOR)

        # already resized
        if np.shape(img) == (IMG_WIDTH, IMG_HEIGHT, 3):
            continue

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)  # resize with ANTI-ALIAS
        img = 5.0 * img - 4.0 * 255.0
        img = np.clip(img, 0.0, 255.0)
        cv2.imwrite(img_full_path, img)

    # load dataset
    dataset_df = dataset_df.sample(frac=1)
    train_loader, test_loader = load_dataset(dataset_df)

    # load or train model
    try:
        print('loading CNN models ...')
        cnn_models = load_cnn_model()

    except Exception as e:
        print(f'CNN model load failed : {e}')
        cnn_models = train_cnn(train_loader)

    # performance evaluation
    _, performance_scores = predict_score_image(test_loader, cnn_models)

    print('PERFORMANCE SCORE :\n')
    print(performance_scores)

    # predict using CNN model
    base_dir = f'{TRAIN_DATA_DIR_PATH}/base'
    sft_generated_dir = f'{TRAIN_DATA_DIR_PATH}/sft_generated_orpo_dataset'

    img_paths_base = [f'{base_dir}/diagram_{i:%06d}.png' for i in range(5)]
    img_paths_sft_generated = [f'{sft_generated_dir}/diagram_{i:%06d}.png' for i in range(50)]
    img_paths = img_paths_base + img_paths_sft_generated

    final_score = predict_score_path(img_paths, cnn_models)

    # save and print prediction result
    log_dir = f'{PROJECT_DIR_PATH}/final_recommend_score/log'
    os.makedirs(log_dir, exist_ok=True)
    final_score.to_csv(f'{log_dir}/final_score.csv')

    print('FINAL PREDICTION :\n')
    print(final_score)
