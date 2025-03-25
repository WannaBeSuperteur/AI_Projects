
import torch.nn as nn


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
    raise NotImplementedError


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


# Auto-Encoder 모델의 Encoder 를 이용한 이미지 인코딩
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
    pass


