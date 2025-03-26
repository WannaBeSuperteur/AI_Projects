import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.io import read_image
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

MAX_EPOCHS = 1000
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
    train_log = {'success': [], 'min_train_loss': [], 'total_epochs': [], 'best_epoch': [], 'elapsed_time (s)': [],
                 'train_loss_list': []}

    auto_encoder.device = device

    # try train until success (threshold : min train loss < 600)
    while True:
        auto_encoder = auto_encoder.to(device)

        # run training
        start_at = time.time()
        train_loss_list, best_epoch_model = train_ae_each_model(auto_encoder, data_loader)
        train_time = round(time.time() - start_at, 2)

        # check train successful & create train log
        min_train_loss = min(train_loss_list)
        is_train_successful = (min_train_loss < 600)
        train_loss_list_ = list(map(lambda x: round(x, 4), train_loss_list))

        train_log['success'].append(is_train_successful)
        train_log['min_train_loss'].append(round(min_train_loss, 4))
        train_log['total_epochs'].append(len(train_loss_list))
        train_log['best_epoch'].append(np.argmin(train_loss_list))
        train_log['elapsed_time (s)'].append(train_time)
        train_log['train_loss_list'].append(train_loss_list_)

        train_log_path = f'{PROJECT_DIR_PATH}/final_recommend_score/log/ae_train_log.csv'
        pd.DataFrame(train_log).to_csv(train_log_path)

        # save and return model if successful
        if is_train_successful:
            ae_encoder = best_epoch_model.encoder
            ae_decoder = best_epoch_model.decoder

            model_path = f'{PROJECT_DIR_PATH}/final_recommend_score/models'
            os.makedirs(model_path, exist_ok=True)

            torch.save(best_epoch_model.state_dict(), f'{model_path}/ae_model.pt')
            torch.save(ae_encoder.state_dict(), f'{model_path}/ae_encoder.pt')
            torch.save(ae_decoder.state_dict(), f'{model_path}/ae_decoder.pt')
            break

        # retry if train failed
        else:
            auto_encoder = define_ae_model()
            auto_encoder.device = device

    return ae_encoder, ae_decoder


# Auto-Encoder 모델 정의
# Create Date : 2025.03.25
# Last Update Date : 2025.03.25
# - learning rate scheduling 에 warm-up 적용

# Arguments:
# - 없음

# Returns:
# - ae_model (nn.Module) : 정의된 Auto-Encoder 모델

def define_ae_model():
    ae_model = UserScoreAE()
    ae_model.optimizer = torch.optim.AdamW(ae_model.parameters(), lr=0.001)

    # 5 epoch 까지는 warm_up, 이후 매 epoch 마다 1.5% 씩 learning rate 감소
    scheduler_lambda = lambda epoch: 0.2 * (epoch + 1) if epoch < 5 else 0.985 ** (epoch - 4)
    ae_model.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=ae_model.optimizer,
                                                           lr_lambda=scheduler_lambda)

    return ae_model


# 각 Auto-Encoder 모델의 학습 실시
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - model       (nn.Module)  : 학습 대상 Auto-Encoder 모델
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

        # test code 처럼 (출력, 이미지, latent vector) 를 출력할 sample (data loader) index 지정
        if current_epoch < 5:
            force_test_idxs = [0, 1, 2, 3, 4]
        elif (current_epoch < 100 and current_epoch % 20 == 0) or current_epoch % 50 == 0:
            force_test_idxs = [0, 1, 2]
        else:
            force_test_idxs = None

        train_loss = run_train_ae(model=model,
                                  train_loader=data_loader,
                                  device=model.device,
                                  loss_func=loss_func,
                                  center_crop=(IMG_HEIGHT // 2, IMG_WIDTH // 2),
                                  force_test_idxs=force_test_idxs)

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

        # stop training if too long
        if current_epoch >= MAX_EPOCHS:
            break

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

    model = UserScoreAEEncoder()
    model_path = f'{PROJECT_DIR_PATH}/final_recommend_score/models/ae_encoder.pt'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    return model


# 학습된 Auto-Encoder 모델의 Encoder 를 이용한 이미지 인코딩
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - ae_encoder  (nn.Module) : Encoder Model
# - image_paths (list(str)) : latent vector 를 출력할 이미지 경로 리스트
# - report_path (str)       : latent vector 출력 결과 및 그 거리에 대한 csv 파일을 저장할 경로

# Returns:
# - {report_path}/ae_test_result_latent_vectors.csv 에 latent vector 출력 결과 저장
# - {report_path}/ae_test_result_latent_vector_distance.csv 에 latent vector 간의 거리 정보 저장

def test_ae_encoder(ae_encoder, image_paths, report_path):

    # latent vector 출력 결과
    img_cnt = len(image_paths)
    latent_vector_dict = {'img_path': image_paths, 'latent_vector': []}
    latent_vector_distance = np.zeros((img_cnt, img_cnt))

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    for img_path in image_paths:
        img_full_path = f'{TRAIN_DATA_DIR_PATH}/{img_path}'
        img_tensor = read_image(img_full_path)
        img_tensor = transform(img_tensor)

        img_tensor = img_tensor.reshape((1, 3, IMG_HEIGHT, IMG_WIDTH))
        img_tensor = img_tensor[:, :, IMG_HEIGHT // 4 : 3 * IMG_HEIGHT // 4, IMG_WIDTH // 4 : 3 * IMG_WIDTH // 4]

        latent_vector = np.array(ae_encoder(img_tensor)[0].detach().cpu())
        latent_vector_dict['latent_vector'].append(latent_vector)

    pd.DataFrame(latent_vector_dict).to_csv(f'{report_path}/ae_test_result_latent_vectors.csv', index=False)

    # latent vector 간의 거리 정보
    latent_vectors = latent_vector_dict['latent_vector']

    for i in range(img_cnt):
        for j in range(img_cnt):
            vec_0 = latent_vectors[i]
            vec_1 = latent_vectors[j]
            latent_vector_distance[i][j] = np.sum(np.square(vec_0 - vec_1))

    latent_vector_dist_df = pd.DataFrame(latent_vector_distance)

    latent_vector_dist_df['img_path'] = latent_vector_dict['img_path']
    latent_vector_dist_df['latent_vector'] = latent_vector_dict['latent_vector']
    latent_vector_dist_df['latent_vector'] = latent_vector_dist_df['latent_vector'].apply(lambda x: np.round(x, 4))

    img_idx_columns = list(latent_vector_dist_df.columns[:img_cnt])
    for img_idx_column in img_idx_columns:
        latent_vector_dist_df[img_idx_column] = latent_vector_dist_df[img_idx_column].apply(lambda x: np.round(x, 4))

    latent_vector_dist_df = latent_vector_dist_df[['img_path', 'latent_vector'] + img_idx_columns]

    latent_vector_dist_df.to_csv(f'{report_path}/ae_test_result_latent_vector_distance.csv', index=False)


if __name__ == '__main__':
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/final_recommend_score/scores.csv')
    log_dir = f'{PROJECT_DIR_PATH}/final_recommend_score/log'

    # resize images to (128, 128)
    img_paths = dataset_df['img_path'].tolist()
    resize_and_normalize_img(img_paths,
                             train_data_dir_path=TRAIN_DATA_DIR_PATH,
                             dest_width=IMG_WIDTH,
                             dest_height=IMG_HEIGHT)

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
    test_img_df = pd.read_csv(f'{PROJECT_DIR_PATH}/final_recommend_score/ae_test_data_list.csv')
    test_img_paths = test_img_df['img_path']

    test_ae_encoder(ae_encoder, image_paths=test_img_paths, report_path=log_dir)
