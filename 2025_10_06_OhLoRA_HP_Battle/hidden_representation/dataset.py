

from torch.utils.data import Dataset, random_split
from torchvision.io import read_image
import torchvision.transforms as transforms

import os
import numpy as np
import pandas as pd


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/datasets'
NUM_CLASSES = 10


base_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


class AutoEncoderImageDataset(Dataset):
    def __init__(self, dataset_df, transform, dataset_name, tvt_type):
        self.img_paths = dataset_df['img_path'].tolist()    # ex: airplane/0001.png
        self.img_nos = dataset_df['img_no'].tolist()
        self.class_idxs = dataset_df['class_idx'].tolist()

        self.transform = transform
        self.dataset_name = dataset_name
        self.tvt_type = tvt_type                            # 'train' or 'test'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = f'{IMAGE_DATA_DIR_PATH}/{self.dataset_name}/{self.tvt_type}/{self.img_paths}'
        image = read_image(img_path)

        # resize and normalize
        image = self.transform(image)
        class_idx = self.class_idxs[idx]
        class_idx_one_hot = np.eye(NUM_CLASSES)[class_idx]

        return image, class_idx_one_hot


# Dataset 생성을 위한 DataFrame 생성
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - tvt_type     (str) : Train/Valid/Test 데이터셋 종류 ('train', 'valid' or 'test')

# Returns:
# - dataset_df (Pandas DataFrame) : Dataset 생성을 위한 DataFrame

def create_dataset_df(dataset_name, tvt_type):
    dataset_dict = {'img_no': [], 'img_path': [], 'class_idx': []}
    dataset_dir_path = f'{PROJECT_DIR_PATH}/datasets/{dataset_name}/{tvt_type}'
    dataset_classes = os.listdir(dataset_dir_path)

    for class_idx, dataset_class in enumerate(dataset_classes):
        dataset_class_dir_path = f'{dataset_dir_path}/{dataset_class}'
        img_names = os.listdir(dataset_class_dir_path)

        for img_name in img_names:
            dataset_dict['img_no'].append(len(dataset_dict['img_no']))
            dataset_dict['img_path'].append(f'{dataset_class_dir_path}/{img_name}')
            dataset_dict['class_idx'].append(class_idx)

    dataset_df = pd.DataFrame(dataset_dict)
    return dataset_df


# Train 데이터셋을 Train 과 Valid 데이터셋으로 분리
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - dataset     (torch.utils.data.Dataset) : train 과 valid 로 분리할 데이터셋
# - train_ratio (float)                    : 학습 데이터의 비율 (default: 0.8)

# Returns:
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - valid_dataset (torch.utils.data.Dataset) : 검증 (valid) 데이터셋

def split_into_train_and_valid(dataset, train_ratio=0.8):
    dataset_size = len(dataset)

    train_size = int(train_ratio * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    return train_dataset, valid_dataset
