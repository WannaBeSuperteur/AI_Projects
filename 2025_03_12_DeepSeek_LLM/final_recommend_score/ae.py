import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.utils import save_image

import pandas as pd
import numpy as np

import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from global_common.torch_training import run_train_ae
from common import resize_and_normalize_img, DiagramImageDataset


torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/training_data'

EARLY_STOPPING_ROUNDS = 10
IMG_HEIGHT = 128
IMG_WIDTH = 128
LATENT_VECTOR_DIM = 32

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4


# Custom Reshape Layer from https://discuss.pytorch.org/t/what-is-reshape-layer-in-pytorch/1110/8
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# Auto-Encoder Conv. Layers for Encoder
class UserScoreAEEncoderConvs(nn.Module):
    def __init__(self):
        super(UserScoreAEEncoderConvs, self).__init__()

        # encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_conv3(x)

        return x


# Auto-Encoder DeConv. Layers for Decoder
class UserScoreAEDecoderDeConvs(nn.Module):
    def __init__(self):
        super(UserScoreAEDecoderDeConvs, self).__init__()

        # decoder
        self.decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU()
        )
        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU()
        )
        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.decoder_deconv1(x)
        x = self.decoder_deconv2(x)
        x = self.decoder_deconv3(x)

        return x


# Encoder of Auto-Encoder
class UserScoreAEEncoder(nn.Module):
    def __init__(self):
        super(UserScoreAEEncoder, self).__init__()

        # encoder
        self.encoder_convs = UserScoreAEEncoderConvs()
        self.encoder_flatten = nn.Flatten()

        self.encoder_fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8 + 512, 512),
            nn.Sigmoid()
        )
        self.encoder_fc2 = nn.Sequential(
            nn.Linear(512, LATENT_VECTOR_DIM),
            nn.Sigmoid()
        )

        # additional fully-connected layer for flatten stream
        self.encoder_fc_flatten_stream = nn.Sequential(
            nn.Linear(3 * IMG_HEIGHT // 2 * IMG_WIDTH // 2, 512),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_conv_stream = self.encoder_convs(x)
        x_conv_stream = self.encoder_flatten(x_conv_stream)

        x_flatten_stream = self.encoder_flatten(x)
        x_flatten_stream = self.encoder_fc_flatten_stream(x_flatten_stream)

        x = torch.cat([x_conv_stream, x_flatten_stream], dim=1)
        x = self.encoder_fc1(x)
        x = self.encoder_fc2(x)

        return x


# Decoder of Auto-Encoder
class UserScoreAEDecoder(nn.Module):
    def __init__(self):
        super(UserScoreAEDecoder, self).__init__()

        # decoder
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(LATENT_VECTOR_DIM, 512),
            nn.Sigmoid()
        )
        self.decoder_fc2 = nn.Sequential(
            nn.Linear(512, 128 * 8 * 8),
            nn.Sigmoid()
        )

        self.decoder_reshape = Reshape(-1, 128, 8, 8)
        self.decoder_deconvs = UserScoreAEDecoderDeConvs()

    def forward(self, x):
        x = self.decoder_fc1(x)
        x = self.decoder_fc2(x)

        x = self.decoder_reshape(x)
        x = self.decoder_deconvs(x)

        return x


# Auto-Encoder
class UserScoreAE(nn.Module):
    def __init__(self):
        super(UserScoreAE, self).__init__()

        # encoder
        self.encoder = UserScoreAEEncoder()
        self.decoder = UserScoreAEDecoder()

    def forward(self, x):
        x = x[:, :, IMG_HEIGHT // 4: 3 * IMG_HEIGHT // 4, IMG_WIDTH // 4: 3 * IMG_WIDTH // 4]
        x = self.encoder(x)
        x = self.decoder(x)

        return x


# 데이터셋 로딩
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터 정보가 저장된 Pandas DataFrame
#                                   columns = ['img_path', 'score']

# Returns:
# - train_loader (DataLoader) : Train 데이터셋을 로딩한 PyTorch DataLoader

def load_dataset(dataset_df):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    train_dataset = DiagramImageDataset(dataset_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    print(f'size of train loader : {len(train_loader.dataset)}')

    # test code
    test_dir_path = f'{PROJECT_DIR_PATH}/final_recommend_score/temp_test_ae'

    for i in range(40):
        img = train_loader.dataset.__getitem__(i)[0]
        os.makedirs(test_dir_path, exist_ok=True)
        save_image(img, f'{test_dir_path}/train_data_{i:02d}.png')

    return train_loader


# Auto-Encoder 모델 학습 실시
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader (Train ONLY)

# Returns:
# - ae_encoder (nn.Module) : Encoder Model
# - ae_decoder (nn.Module) : Decoder Model
# - log/ae_train_log.csv 파일에 학습 로그 저장

def train_ae(data_loader):

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training : {device}')

    if 'cuda' in str(device):
        print(f'cuda is available with device {torch.cuda.get_device_name()}')

    auto_encoder = define_ae_model()
    summary(auto_encoder, input_size=(TRAIN_BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH))

    # for train log
    os.makedirs(f'{PROJECT_DIR_PATH}/final_recommend_score/log', exist_ok=True)
    train_log = {'min_train_loss': [], 'total_epochs': [], 'best_epoch': [], 'elapsed_time (s)': [],
                 'train_loss_list': []}

    # train
    auto_encoder.device = device

    start_at = time.time()
    train_loss_list, best_epoch_model = train_ae_each_model(auto_encoder, data_loader)
    train_time = round(time.time() - start_at, 2)

    train_log['min_train_loss'].append(round(min(train_loss_list), 4))
    train_log['totel_epochs'].append(len(train_loss_list))
    train_log['best_epoch'].append(np.argmin(train_loss_list))
    train_log['elapsed_time (s)'].append(train_time)
    train_log['train_loss_list'].append(train_loss_list)

    # save and return model
    ae_encoder = best_epoch_model.encoder
    ae_decoder = best_epoch_model.decoder

    model_path = f'{PROJECT_DIR_PATH}/final_recommend_score/models'
    os.makedirs(model_path, exist_ok=True)

    torch.save(best_epoch_model.state_dict(), f'{model_path}/ae_model.pt')
    torch.save(ae_encoder.state_dict(), f'{model_path}/ae_encoder.pt')
    torch.save(ae_decoder.state_dict(), f'{model_path}/ae_decoder.pt')

    return ae_encoder, ae_decoder


# Auto-Encoder 모델 정의
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - ae_model (nn.Module) : 정의된 Auto-Encoder 모델

def define_ae_model():
    ae_model = UserScoreAE()
    ae_model.optimizer = torch.optim.AdamW(ae_model.parameters(), lr=0.001)
    ae_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=ae_model.optimizer,
                                                                    T_max=10,
                                                                    eta_min=0)

    return ae_model


# 각 Auto-Encoder 모델의 학습 실시
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - model       (nn.Module)  : 학습 대상 CNN 모델
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader (Train ONLY)

# Returns:
# - train_loss_list  (list(float)) : train loss 기록
# - best_epoch_model (nn.Module)   : 가장 낮은 train loss 에서의 Auto-Encoder 모델

def train_ae_each_model(model, data_loader):
    current_epoch = 0
    min_train_loss_epoch = -1  # Loss-based Early Stopping
    min_train_loss = None
    best_epoch_model = None

    train_loss_list = []

    # loss function
    loss_func = nn.MSELoss(reduction='sum')

    while True:
        train_loss = run_train_ae(model=model,
                                  train_loader=data_loader,
                                  device=model.device,
                                  loss_func=loss_func,
                                  center_crop=(IMG_HEIGHT // 2, IMG_WIDTH // 2))

        print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}')
        train_loss_list.append(train_loss)

        model.scheduler.step()

        if min_train_loss is None or train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_loss_epoch = current_epoch

            best_epoch_model = UserScoreAE().to(model.device)
            best_epoch_model.load_state_dict(model.state_dict())

        if current_epoch - min_train_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return train_loss_list, best_epoch_model


# Auto-Encoder 모델의 Encoder 불러오기
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - ae_encoder (nn.Module) : load 된 Auto-Encoder 모델의 Encoder

def load_ae_encoder():

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for loading model : {device}')

    model = UserScoreAE()
    model_path = f'{PROJECT_DIR_PATH}/final_recommend_score/models/ae_encoder.pt'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    return model


# 학습된 Auto-Encoder 모델의 Encoder 를 이용한 이미지 인코딩
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - data_loader (DataLoader) : 인코딩할 입력 데이터셋을 로딩한 PyTorch DataLoader
# - ae_encoder  (nn.Module)  : Encoder Model

# Returns:
# - encoded_vector (np.array) : Encoder 에 의해 인코딩된 벡터

def encode_image(data_loader, ae_encoder):
    raise NotImplementedError


if __name__ == '__main__':
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/final_recommend_score/scores.csv')
    log_dir = f'{PROJECT_DIR_PATH}/final_recommend_score/log'

    # resize images to (128, 128)
    img_paths = dataset_df['img_path'].tolist()
    resize_and_normalize_img(img_paths,
                             train_data_dir_path=TRAIN_DATA_DIR_PATH,
                             img_width=IMG_WIDTH,
                             img_height=IMG_HEIGHT)

    # load dataset
    dataset_df = dataset_df.sample(frac=1, random_state=20250325)  # shuffle image sample order
    train_loader = load_dataset(dataset_df)

    # load or train model
    try:
        print('loading Auto-Encoder models ...')
        ae_encoder = load_ae_encoder()
        print('loading Auto-Encoder models successful!')

    except Exception as e:
        print(f'Auto-Encoder model load failed : {e}')
        ae_encoder, _ = train_ae(train_loader)

    # performance evaluation
    report_path = f'{log_dir}/ae_test_result.csv'

    # TODO implement
