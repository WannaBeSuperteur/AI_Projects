
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import os
import pandas as pd

IMG_HEIGHT = 256
IMG_WIDTH = 256

base_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
TRAIN_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results'
TRAIN_BATCH_SIZE = 16


class CNNImageDataset(Dataset):
    def __init__(self, dataset_df, transform, property_name):
        self.img_paths = dataset_df['img_path'].tolist()
        self.img_nos = dataset_df['img_no'].tolist()
        self.gender_scores = dataset_df['gender'].tolist()
        self.quality_scores = dataset_df['quality'].tolist()

        self.transform = transform
        self.property_name = property_name

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx, ):
        img_path = f'{TRAIN_DATA_DIR_PATH}/{self.img_nos[idx]:06d}.jpg'
        image = read_image(img_path)

        # resize and normalize
        image = self.transform(image)

        if self.property_name == 'gender':
            score = self.gender_scores[idx]
        elif self.property_name == 'quality':
            score = self.quality_scores[idx]
        else:
            raise Exception("property_name must be one of ['gender', 'quality'].")

        return image, score


# Original StyleGAN 이 생성한 이미지 중 첫 2,000 장의 데이터셋 생성을 위한 정보가 있는 Pandas DataFrame 생성
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - labeled_df (Pandas DataFrame) : 2,000 장의 데이터를 train data 로 하는 데이터셋 생성을 위한 Pandas DataFrame
#                                   columns = ['img_path', 'img_no', 'gender', 'quality']

def create_train_dataset_df():
    labeled_data_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/synthesize_results_quality_and_gender.csv'
    labeled_df = pd.read_csv(labeled_data_path)

    labeled_df['img_no'] = labeled_df['img_name']
    labeled_df['img_path'] = labeled_df['img_name'].apply(lambda x: f'{TRAIN_DATA_DIR_PATH}/{x:06d}.jpg')

    labeled_df.drop(columns=['img_name'], inplace=True)

    return labeled_df


# Original StyleGAN 이 생성한 이미지 중 첫 2,000 장의 데이터셋 로딩
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name (str) : 핵심 속성 값 이름 ('gender' or 'quality')

# Returns:
# - data_loader (DataLoader) : 2,000 장의 데이터를 train data 로 하는 DataLoader

def load_dataset(property_name):
    dataset_df = create_train_dataset_df()
    dataset = CNNImageDataset(dataset_df, transform=base_transform, property_name=property_name)

    data_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    return data_loader


# Original StyleGAN 이 생성한 이미지 중 나머지 8,000 장 로딩
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name (str) : 핵심 속성 값 이름 ('gender' or 'quality')

# Returns:
# - data_loader (DataLoader) : 나머지 8,000 장의 데이터를 test (inference) data 로 하는 DataLoader

def load_remaining_images_dataset(property_name):
    raise NotImplementedError


# CNN 모델 정의
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name (str) : 핵심 속성 값 이름 ('gender' or 'quality')

# Returns:
# - cnn_model (nn.Module) : 학습할 CNN 모델

def define_cnn_model(property_name):
    raise NotImplementedError


# 모델 학습 실시 (K-Fold Cross Validation)
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - data_loader (DataLoader) : 2,000 장의 데이터를 train data 로 하는 DataLoader

# Returns:
# - cnn_models (list(nn.Module)) : 학습된 CNN Model 의 리스트 (총 K 개의 모델)
# - log/cnn_train_log.csv 파일에 학습 로그 저장

def train_cnn_models(data_loader):
    raise NotImplementedError


# 각 모델 (각각의 Fold 에 대한) 의 학습 실시
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - model       (nn.Module)  : 학습 대상 CNN 모델
# - data_loader (DataLoader) : 2,000 장의 데이터를 train data 로 하는 DataLoader
# - train_idxs  (np.array)   : 학습 데이터로 지정된 인덱스
# - valid_idxs  (np.array)   : 검증 데이터로 지정된 인덱스

# Returns:
# - val_loss_list    (list(float)) : valid loss 기록
# - best_epoch_model (nn.Module)   : 가장 낮은 valid loss 에서의 CNN 모델

def train_cnn_each_model(model, data_loader, train_idxs, valid_idxs):
    raise NotImplementedError


# 학습된 CNN 모델 불러오기
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name (str) : 핵심 속성 값 이름 ('gender' or 'quality')

# Returns:
# - cnn_models (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

def load_cnn_model(property_name):
    raise NotImplementedError


# 학습된 모델을 이용하여 나머지 8,000 장의 이미지에 대해 핵심 속성 값 예측 (Ensemble 의 아이디어 / K 개 모델의 평균으로)
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name           (str)             : 핵심 속성 값 이름 ('gender' or 'quality')
# - remaining_images_loader (DataLoader)      : 나머지 8,000 장의 이미지 데이터셋을 로딩한 PyTorch DataLoader
# - cnn_models              (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)
# - report_path             (str)             : final_score 의 report 를 저장할 경로

# Returns:
# - final_score (Pandas DataFrame) : 해당 속성 값에 대한 모델 예측값을 저장한 Pandas DataFrame
#                                    columns = ['img_no', 'img_path', 'property_{property_name}_final_score',
#                                               'score_model_0', 'score_model_1', ...]

def predict_score_remaining_images(property_name, remaining_images_loader, cnn_models, report_path):
    raise NotImplementedError
