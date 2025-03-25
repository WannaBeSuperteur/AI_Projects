import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4


class UserScoreAE(nn.Module):
    def __init__(self):
        super(UserScoreAE, self).__init__()

    def forward(self, x):
        return 0  # temp


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
    raise NotImplementedError


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
        cnn_models = load_ae_encoder()
        print('loading Auto-Encoder models successful!')

    except Exception as e:
        print(f'Auto-Encoder model load failed : {e}')
        cnn_models = train_ae(train_loader)

    # performance evaluation
    report_path = f'{log_dir}/ae_test_result.csv'

    # TODO implement
