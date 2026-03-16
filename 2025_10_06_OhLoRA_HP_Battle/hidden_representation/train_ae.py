
import torch
from torch.utils.data import DataLoader

from auto_encoder import AutoEncoder_1_28_28, AutoEncoder_3_32_32
from dataset import create_dataset_df
from dataset import base_transform, AutoEncoderImageDataset

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from global_common.torch_training import run_train_ae


TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, TEST_BATCH_SIZE = 16, 4, 4
EARLY_STOPPING_ROUNDS = 10
MAX_EPOCHS = 1000
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


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

# Returns:
# - train_loss_list  (list)             : train loss 의 list
# - best_epoch_model (torch.nn.modules) : Loss 가 가장 낮은 epoch 에서의 Auto-Encoder 모델

def train_ae(ae_model, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    train_loss_list = []

    current_epoch = 0
    min_train_loss_epoch = -1  # Loss-based Early Stopping
    min_train_loss = None
    best_epoch_model = None

    while True:

        # test code 처럼 (출력, 이미지, latent vector) 를 출력할 sample (data loader) index 지정
        if current_epoch == 0:
            force_test_idxs = [0, 100, 500, 1000, 2000, 3000]
        else:
            force_test_idxs = None

        train_loss = run_train_ae(model=ae_model,
                                  train_loader=train_loader,
                                  device=ae_model.device,
                                  force_test_idxs=force_test_idxs)

        if min_train_loss is None:
            print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}')
        else:
            print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}, min_train_loss : {min_train_loss:.4f}')
        train_loss_list.append(train_loss)

        ae_model.scheduler.step()

        if min_train_loss is None or train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_loss_epoch = current_epoch

            if dataset_name == 'cifar_10':
                best_epoch_model = AutoEncoder_3_32_32().to(ae_model.device)
            else:
                best_epoch_model = AutoEncoder_1_28_28().to(ae_model.device)
            best_epoch_model.load_state_dict(ae_model.state_dict())

        if current_epoch - min_train_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

        # stop training if too long
        if current_epoch >= MAX_EPOCHS:
            break

    return train_loss_list, best_epoch_model


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        print(f'\n==== DATASET: {dataset_name} ====\n')

        ae_model = load_ae_model_before_train(dataset_name)
        train_dataset, test_dataset = load_dataset(dataset_name)
#        train_train_dataset, train_valid_dataset = split_into_train_and_valid(train_dataset)

        train_loss_list, best_epoch_model = train_ae(ae_model, train_dataset)
#        test_ae(ae_model, test_dataset)

        # save encoder and decoder
        ae_encoder = best_epoch_model.encoder
        ae_decoder = best_epoch_model.decoder

        model_path = f'{PROJECT_DIR_PATH}/models'
        os.makedirs(model_path, exist_ok=True)

        torch.save(best_epoch_model.state_dict(), f'{model_path}/ae_model_{dataset_name}.pt')
        torch.save(ae_encoder.state_dict(), f'{model_path}/ae_encoder_{dataset_name}.pt')
        torch.save(ae_decoder.state_dict(), f'{model_path}/ae_decoder_{dataset_name}.pt')
