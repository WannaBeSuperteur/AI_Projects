import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.utils import save_image

import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from common import resize_and_normalize_img, DiagramImageDataset

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/training_data'

EARLY_STOPPING_ROUNDS = 10
IMG_HEIGHT = 128
IMG_WIDTH = 128
HIDDEN_DIM = 32

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4


# Custom Reshape Layer from https://discuss.pytorch.org/t/what-is-reshape-layer-in-pytorch/1110/8
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# Auto-Encoder
class UserScoreAE(nn.Module):
    def __init__(self):
        super(UserScoreAE, self).__init__()

        # encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.encoder_conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.encoder_flatten = nn.Flatten()
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.encoder_fc2 = nn.Sequential(
            nn.Linear(512, HIDDEN_DIM),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.encoder = nn.Sequential(
            self.encoder_conv1,
            self.encoder_conv2,
            self.encoder_conv3,
            self.encoder_conv4,
            self.encoder_conv5,
            self.encoder_flatten,
            self.encoder_fc1,
            self.encoder_fc2
        )

        # decoder
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 512),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.decoder_fc2 = nn.Sequential(
            nn.Linear(512, 256 * 4 * 4),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.decoder_reshape = Reshape(-1, 256, 4, 4)
        self.decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.decoder_deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.decoder_deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15)
        )
        self.decoder = nn.Sequential(
            self.decoder_fc1,
            self.decoder_fc2,
            self.decoder_reshape,
            self.decoder_deconv1,
            self.decoder_deconv2,
            self.decoder_deconv3,
            self.decoder_deconv4,
            self.decoder_deconv5
        )

    def forward(self, x):
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
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader

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

    raise NotImplementedError


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
