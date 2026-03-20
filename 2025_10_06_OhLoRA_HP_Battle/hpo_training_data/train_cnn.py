
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, f1_score

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from global_common.torch_training import run_train, run_validation


TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, TEST_BATCH_SIZE = 16, 4, 4
EARLY_STOPPING_ROUNDS = 10
MAX_EPOCHS = 70

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

        self.dropout_conv_earlier = dropout_conv_earlier
        self.dropout_conv_later = dropout_conv_later
        self.dropout_fc = dropout_fc
        self.activation_func = activation_func

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
            nn.Linear(512, NUM_CLASSES),
            nn.Softmax(dim=1)
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

        self.dropout_conv_earlier = dropout_conv_earlier
        self.dropout_conv_later = dropout_conv_later
        self.dropout_fc = dropout_fc
        self.activation_func = activation_func

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
            nn.Linear(512, NUM_CLASSES),
            nn.Softmax(dim=1)
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
# Last Update Date : 2026.03.20
# - 학습 데이터셋 -> 학습 + 검증 데이터셋으로 분리
# - 학습 및 테스트 데이터셋 분포 정보 반환 추가
# - TOP 5 label 데이터셋만 학습/테스트하도록 설정

# Arguments:
# - dataset_name (str)  : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - constraints  (dict) : 데이터셋 constraint list

# Returns:
# - train_dataset               (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - valid_dataset               (torch.utils.data.Dataset) : 검증 (valid) 데이터셋
# - test_dataset                (torch.utils.data.Dataset) : 테스트 데이터셋
# - train_dataset_label_distrib (list)                     : 학습+검증 데이터셋 label 분포
# - test_dataset_label_distrib  (list)                     : 테스트 데이터셋 label 분포

def load_dataset(dataset_name, constraints):
    dataset_info_df = pd.read_csv(f'test/{dataset_name}/dataset_info.csv', index_col=0)

    for constraint_col in constraints.keys():
        left = constraints[constraint_col][0]
        right = constraints[constraint_col][1]
        dataset_info_df = dataset_info_df[dataset_info_df[constraint_col].between(left, right)]

    train_dataset_info_df = dataset_info_df[dataset_info_df['tvt_type'] == 'train']
    test_dataset_info_df = dataset_info_df[dataset_info_df['tvt_type'] == 'test']

    # train/valid dataset
    train_dataset_value_counts = train_dataset_info_df['label'].value_counts()
    top_5_labels = list(train_dataset_value_counts[:5].index)

    train_dataset_info_df = train_dataset_info_df[train_dataset_info_df['label'].isin(top_5_labels)]
    train_valid_dataset = BaseCNNImageDataset(train_dataset_info_df,
                                              transform=cnn_base_transform,
                                              dataset_name=dataset_name,
                                              tvt_type='train')

    n_train_size = int(0.85 * len(train_valid_dataset))
    n_valid_size = len(train_valid_dataset) - n_train_size
    train_dataset, valid_dataset = random_split(train_valid_dataset, [n_train_size, n_valid_size])

    # test dataset
    test_dataset_info_df = test_dataset_info_df[test_dataset_info_df['label'].isin(top_5_labels)]
    test_dataset = BaseCNNImageDataset(test_dataset_info_df,
                                       transform=cnn_base_transform,
                                       dataset_name=dataset_name,
                                       tvt_type='test')

    # data label distribution
    train_dataset_label_distrib = list(train_dataset_info_df['label'].value_counts())
    test_dataset_label_distrib = list(test_dataset_info_df['label'].value_counts())
    train_dataset_label_distrib.sort(reverse=True)
    test_dataset_label_distrib.sort(reverse=True)

    return train_dataset, valid_dataset, test_dataset, train_dataset_label_distrib, test_dataset_label_distrib


# Base CNN 학습 실시 및 모델 저장
# Create Date : 2026.03.20
# Last Update Date : -

# Arguments:
# - cnn_model     (nn.Module)                : 학습할 CNN 모델
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - valid_dataset (torch.utils.data.Dataset) : 검증 (valid) 데이터셋

# Returns:
# - train_loss_list  (list)             : train loss 의 list
# - best_epoch_model (torch.nn.modules) : Loss 가 가장 낮은 epoch 에서의 Auto-Encoder 모델

def train_cnn(cnn_model, train_dataset, valid_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model = None
    val_loss_list = []

    while True:
        train_loss = run_train(model=cnn_model,
                               train_loader=train_loader,
                               device=cnn_model.device,
                               unsqueeze_label=False)

        _, val_loss = run_validation(model=cnn_model,
                                     valid_loader=valid_loader,
                                     device=cnn_model.device,
                                     unsqueeze_label=False)

        print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}, val_loss : {val_loss:.4f}')

        val_loss_cpu = float(val_loss.detach().cpu())
        val_loss_list.append(val_loss_cpu)

        cnn_model.scheduler.step()

        if min_val_loss is None or val_loss < min_val_loss - 1e-5:
            min_val_loss = val_loss
            min_val_loss_epoch = current_epoch

            best_epoch_model = cnn_model.__class__(dropout_conv_earlier=cnn_model.dropout_conv_earlier,
                                                   dropout_conv_later=cnn_model.dropout_conv_later,
                                                   dropout_fc=cnn_model.dropout_fc,
                                                   activation_func=cnn_model.activation_func).to(cnn_model.device)
            best_epoch_model.load_state_dict(cnn_model.state_dict())
            best_epoch_model.device = cnn_model.device
            best_epoch_model.to(cnn_model.device)

        if current_epoch - min_val_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        if current_epoch >= MAX_EPOCHS:
            break

        current_epoch += 1

    return val_loss_list, best_epoch_model


# Base CNN 테스트
# Create Date : 2026.03.20
# Last Update Date : -

# Arguments:
# - cnn_model       (nn.Module)                : 학습할 CNN 모델
# - test_dataset    (torch.utils.data.Dataset) : 테스트 데이터셋
# - print_cf_matrix (bool)                     : Confusion Matrix 출력 여부

# Returns:
# - accuracy        (float) : 테스트 결과 정확도
# - f1_score_macro  (float) : 테스트 결과 Macro F1 Score
# - f1_score_micro  (float) : 테스트 결과 Micro F1 Score

def test_cnn(cnn_model, test_dataset, print_cf_matrix=False):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    all_pred = np.array([])
    all_gt = np.array([])
    cf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(cnn_model.device)
        predictions = cnn_model(images)
        predictions_np = predictions.detach().cpu().numpy()
        labels_np = labels.numpy()

        pred = np.argmax(predictions_np, axis=1)
        gt = np.argmax(labels_np, axis=1)

        for pred_item, gt_item in zip(pred, gt):
            cf_matrix[pred_item][gt_item] += 1

        all_pred = np.concatenate((all_pred, pred))
        all_gt = np.concatenate((all_gt, gt))

    accuracy = round(accuracy_score(all_gt, all_pred), 4)
    f1_score_macro = round(f1_score(all_gt, all_pred, average='macro'), 4)
    f1_score_micro = round(f1_score(all_gt, all_pred, average='micro'), 4)

    if print_cf_matrix:
        print('==== CONFUSION MATRIX ====')
        print(cf_matrix)

    return accuracy, f1_score_macro, f1_score_micro


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    hps = {'dropout_conv_earlier': 0.05,
           'dropout_conv_later': 0.05,
           'dropout_fc': 0.45,
           'lr': 0.0001,
           'activation_func': 'leaky_relu',
           'optimizer': 'adamw',
           'scheduler': 'exp_95'}

    for dataset_name in dataset_names:
        print(f'\n==== DATASET: {dataset_name} ====\n')

        if dataset_name == 'cifar_10':
            constraints = {'value': [160, 255]}
        elif dataset_name == 'fashion_mnist':
            constraints = {'value': [56, 64]}
        else:  # mnist
            constraints = {'value': [40, 64]}

        cnn_model = load_cnn_model_before_train(dataset_name, hps)
        train_dataset, valid_dataset, test_dataset, train_dataset_label_distrib, test_dataset_label_distrib =(
            load_dataset(dataset_name, constraints))

        print(f'train data label distrib : {train_dataset_label_distrib}')
        print(f'test data label distrib : {test_dataset_label_distrib}')

        val_loss_list, best_epoch_model = train_cnn(cnn_model, train_dataset, valid_dataset)
        accuracy, f1_score_macro, f1_score_micro = test_cnn(best_epoch_model, test_dataset, print_cf_matrix=True)

        print(f'accuracy : {accuracy}, f1_score: (macro: {f1_score_macro}, micro: {f1_score_micro})')
