
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms


TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, TEST_BATCH_SIZE = 16, 4, 4
EARLY_STOPPING_ROUNDS = 10
MAX_EPOCHS = 300

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/datasets'

NUM_CLASSES = 10
LABELS_CIFAR_10 = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                   'ship': 8, 'truck': 9}

cnn_base_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# for 'fashion_mnist' and 'mnist' dataset
class BaseCNN_1_28_28(nn.Module):
    def get_conv_activation_func(self, activation_func):
        if activation_func == 'leaky_relu':
            return nn.LeakyReLU
        elif activation_func == 'relu':
            return nn.ReLU

    def __init__(self, dropout_conv_earlier, dropout_conv_later, dropout_fc, activation_func):
        super(BaseCNN_1_28_28, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.Tanh(),
            nn.Dropout(dropout_fc)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  # 26 x 26
        x = self.conv2(x)  # 24 x 24
        x = self.pool1(x)  # 12 x 12

        x = self.conv3(x)  # 10 x 10
        x = self.conv4(x)  # 8 x 8
        x = self.conv5(x)  # 6 x 6

        x = x.view(-1, 128 * 6 * 6)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# for 'cifar-10' dataset
class BaseCNN_3_32_32(nn.Module):
    def get_conv_activation_func(self, activation_func):
        if activation_func == 'leaky_relu':
            return nn.LeakyReLU
        elif activation_func == 'relu':
            return nn.ReLU

    def __init__(self, dropout_conv_earlier, dropout_conv_later, dropout_fc, activation_func):
        super(BaseCNN_3_32_32, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.Tanh(),
            nn.Dropout(dropout_fc)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  # 30 x 30
        x = self.conv2(x)  # 28 x 28
        x = self.pool1(x)  # 14 x 14

        x = self.conv3(x)  # 12 x 12
        x = self.pool2(x)  # 6 x 6

        x = self.conv4(x)  # 4 x 4

        x = x.view(-1, 128 * 4 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# Base CNN 모델 학습/테스트 데이터셋
class BaseCNNImageDataset(Dataset):
    def __init__(self, dataset_df, transform, dataset_name, tvt_type):
        self.img_paths = dataset_df['img_path'].tolist()  # ex: airplane/0001.png
        self.labels = dataset_df['label'].tolist()

        self.transform = transform
        self.dataset_name = dataset_name
        self.tvt_type = tvt_type                          # 'train' or 'test'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = f'{IMAGE_DATA_DIR_PATH}/{self.dataset_name}/{self.tvt_type}/{self.img_paths[idx]}'

        if self.dataset_name == 'cifar_10':
            image = read_image(img_path)
        else:
            image = read_image(img_path, mode=ImageReadMode.GRAY)

        # resize and normalize
        image = self.transform(image)

        # class index
        if self.dataset_name == 'cifar_10':
            class_idx = LABELS_CIFAR_10[self.labels[idx]]
        else:
            class_idx = int(self.labels[idx])
        class_idx_one_hot = np.eye(NUM_CLASSES)[class_idx]

        return image, class_idx_one_hot


# Base CNN 모델 로딩 (학습 전의 모델 로딩 -> 이후 학습 실시)
# Create Date : 2026.03.19
# Last Update Date : -

# Arguments:
# - dataset_name (str)  : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - hps          (dict) : 하이퍼파라미터 목록

# Returns:
# - cnn_model (nn.Module) : Auto-Encoder 모델

def load_cnn_model_before_train(dataset_name, hps):
    if dataset_name == 'cifar_10':
        cnn_model = BaseCNN_3_32_32(dropout_conv_earlier=hps['dropout_conv_earlier'],
                                    dropout_conv_later=hps['dropout_conv_later'],
                                    dropout_fc=hps['dropout_fc'],
                                    activation_func=hps['activation_func'])
    else:
        cnn_model = BaseCNN_1_28_28(dropout_conv_earlier=hps['dropout_conv_earlier'],
                                    dropout_conv_later=hps['dropout_conv_later'],
                                    dropout_fc=hps['dropout_fc'],
                                    activation_func=hps['activation_func'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model.to(device)
    cnn_model.device = device

    # optimizer
    assert hps['optimizer'] in ['adam', 'adamw']

    if hps['optimizer'] == 'adam':
        cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(),
                                                lr=hps['lr'])
    elif hps['optimizer'] == 'adamw':
        cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(),
                                                lr=hps['lr'])

    # learning rate scheduler
    assert hps['scheduler'] in ['exp_90', 'exp_95', 'exp_98', 'cosine']

    if hps['scheduler'] == 'exp_90':
        cnn_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cnn_model.optimizer,
                                                                     gamma=0.9)
    elif hps['scheduler'] == 'exp_95':
        cnn_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cnn_model.optimizer,
                                                                     gamma=0.95)
    elif hps['scheduler'] == 'exp_98':
        cnn_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cnn_model.optimizer,
                                                                     gamma=0.98)
    elif hps['scheduler'] == 'cosine':
        cnn_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=cnn_model.optimizer,
                                                                         T_max=10,
                                                                         eta_min=0)

    return cnn_model


# Base CNN 학습용 데이터셋 로딩
# Create Date : 2026.03.19
# Last Update Date : -

# Arguments:
# - dataset_name (str)  : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - constraints  (dict) : 데이터셋 constraint list

# Returns:
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - test_dataset  (torch.utils.data.Dataset) : 테스트 데이터셋

def load_dataset(dataset_name, constraints):
    dataset_info_df = pd.read_csv(f'test/{dataset_name}/dataset_info.csv', index_col=0)

    for constraint_col in constraints.keys():
        left = constraints[constraint_col][0]
        right = constraints[constraint_col][1]
        dataset_info_df = dataset_info_df[dataset_info_df[constraint_col].between(left, right)]

    train_dataset_info_df = dataset_info_df[dataset_info_df['tvt_type'] == 'train']
    test_dataset_info_df = dataset_info_df[dataset_info_df['tvt_type'] == 'test']

    train_dataset = BaseCNNImageDataset(train_dataset_info_df,
                                        transform=cnn_base_transform,
                                        dataset_name=dataset_name,
                                        tvt_type='train')

    test_dataset = BaseCNNImageDataset(test_dataset_info_df,
                                       transform=cnn_base_transform,
                                       dataset_name=dataset_name,
                                       tvt_type='test')

    return train_dataset, test_dataset


# Base CNN 학습 실시 및 모델 저장
# Create Date : 2026.03.20
# Last Update Date : -

# Arguments:
# - cnn_model     (nn.Module)                : 학습할 CNN 모델
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋

# Returns:
# - train_loss_list  (list)             : train loss 의 list
# - best_epoch_model (torch.nn.modules) : Loss 가 가장 낮은 epoch 에서의 Auto-Encoder 모델

def train_cnn(cnn_model, train_dataset):
    raise NotImplementedError


# Base CNN 테스트
# Create Date : 2026.03.20
# Last Update Date : -

# Arguments:
# - cnn_model    (nn.Module)                : 학습할 CNN 모델
# - test_dataset (torch.utils.data.Dataset) : 테스트 데이터셋

# Returns:
# - accuracy (float) : 테스트 결과 정확도
# - f1_score (float) : 테스트 결과 F1 Score

def test_cnn(cnn_model, train_dataset):
    raise NotImplementedError


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    hps = {'dropout_conv_earlier': 0.05,
           'dropout_conv_later': 0.05,
           'dropout_fc': 0.45,
           'lr': 0.001,
           'activation_func': 'leaky_relu',
           'optimizer': 'adamw',
           'scheduler': 'exp_95'}

    for dataset_name in dataset_names:
        print(f'\n==== DATASET: {dataset_name} ====\n')

        if dataset_name == 'cifar_10':
            constraints = {'value': [224, 255]}
        else:
            constraints = {'value': [56, 64]}

        cnn_model = load_cnn_model_before_train(dataset_name, hps)
        train_dataset, test_dataset = load_dataset(dataset_name, constraints)

        train_loss_list, best_epoch_model = train_cnn(cnn_model, train_dataset)
        accuracy, f1_score = test_cnn(cnn_model, test_dataset)

        # save encoder and decoder
        ae_encoder = best_epoch_model.encoder
        ae_decoder = best_epoch_model.decoder

        model_path = f'{PROJECT_DIR_PATH}/models'
        os.makedirs(model_path, exist_ok=True)

        torch.save(best_epoch_model.state_dict(), f'{model_path}/ae_model_{dataset_name}.pt')
        torch.save(ae_encoder.state_dict(), f'{model_path}/ae_encoder_{dataset_name}.pt')
        torch.save(ae_decoder.state_dict(), f'{model_path}/ae_decoder_{dataset_name}.pt')
