
import torch
from torch.utils.data import DataLoader

from auto_encoder import AutoEncoder_1_28_28, AutoEncoder_3_32_32
from dataset import split_into_train_and_valid, create_dataset_df
from dataset import base_transform, AutoEncoderImageDataset

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from global_common.torch_training import run_train_ae


TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, TEST_BATCH_SIZE = 16, 4, 4


# Auto Encoder 모델 로딩 (학습 전의 모델 로딩 -> 이후 학습 실시)
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - ae_model (torch.nn.modules) : Auto-Encoder 모델

def load_ae_model_before_train(dataset_name):
    if dataset_name == 'cifar_10':
        ae_model = AutoEncoder_3_32_32()
    else:
        ae_model = AutoEncoder_1_28_28()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_model.to(device)
    ae_model.device = device
    ae_model.optimizer = torch.optim.AdamW(ae_model.parameters(), lr=0.001)
    ae_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=ae_model.optimizer, gamma=0.95)

    return ae_model


# Auto Encoder 학습용 데이터셋 로딩
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - test_dataset  (torch.utils.data.Dataset) : 테스트 데이터셋

def load_dataset(dataset_name):
    train_dataset_df = create_dataset_df(dataset_name, tvt_type='train')
    test_dataset_df = create_dataset_df(dataset_name, tvt_type='test')

    train_dataset = AutoEncoderImageDataset(train_dataset_df,
                                            transform=base_transform,
                                            dataset_name=dataset_name,
                                            tvt_type='train')

    test_dataset = AutoEncoderImageDataset(test_dataset_df,
                                           transform=base_transform,
                                           dataset_name=dataset_name,
                                           tvt_type='test')

    return train_dataset, test_dataset


# Auto Encoder 학습 실시 및 모델 저장
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - ae_model      (torch.nn.modules)         : Auto-Encoder 모델
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋

def train_ae(ae_model, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    while True:
        train_loss = run_train_ae(model=ae_model, train_loader=train_loader, device=ae_model.device)
        print(train_loss)
        raise NotImplementedError


# Auto Encoder 테스트 실시 및 테스트 결과 저장
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - ae_model     (torch.nn.modules)         : Auto-Encoder 모델
# - test_dataset (torch.utils.data.Dataset) : 테스트 데이터

def test_ae(ae_model, test_dataset):
    raise NotImplementedError


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        ae_model = load_ae_model_before_train(dataset_name)
        train_dataset, test_dataset = load_dataset(dataset_name)
        train_train_dataset, train_valid_dataset = split_into_train_and_valid(train_dataset)

        train_ae(ae_model, train_dataset)
        test_ae(ae_model, test_dataset)
