import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/training_data'


class DiagramImageDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_paths = dataset_df['img_path'].tolist()
        self.scores = dataset_df['score'].tolist()
        self.transform = transform

    def __len__(self):
        return 8 * len(self.img_paths)

    def __getitem__(self, idx):
        true_idx = idx // (4 * 2)
        rotate_angle = idx % 4        # 0 (원본), 1 (반시계 90도), 2 (180도), 3 (시계 90도) 회전
        flip_option = (idx // 4) % 2  # 0 (no flip), 1 (vertical flip)

        img_path = f'{TRAIN_DATA_DIR_PATH}/{self.img_paths[true_idx]}'
        image = read_image(img_path)

        # resize and normalize
        image = self.transform(image)
        score = self.scores[true_idx] / 5.0

        # rotate
        if rotate_angle == 1:
            image = v2.functional.rotate(image, 90)

        elif rotate_angle == 2:
            image = v2.functional.rotate(image, 180)

        elif rotate_angle == 3:
            image = v2.functional.rotate(image, 270)

        # flip
        if flip_option == 1:
            image = v2.functional.vertical_flip(image)

        return image, score


# 원본 Diagram 을 128 x 128 resize 및 정규화 (각 픽셀의 "흰색보다 어두운 정도"를 5배로 하여 이미지 밝기를 어둡게 조정)
# Create Date : 2025.03.25
# Last Update Date : -

# Arguments:
# - img_paths           (list(str)) : 학습 데이터가 있는 이미지 경로의 리스트
# - train_data_dir_path (str)       : 학습 데이터가 있는 디렉토리 경로
# - img_width           (int)       : 이미지 가로 길이
# - img_height          (int)       : 이미지 세로 길이

# Returns:
# - img_paths 의 Diagram 이미지를 128 x 128 resize 및 정규화 실시

def resize_and_normalize_img(img_paths, train_data_dir_path, img_width, img_height):
    for idx, img_path in enumerate(img_paths):
        if idx % 100 == 0:
            print(f'resizing diagram image progress : {idx}')

        img_full_path = f'{train_data_dir_path}/{img_path}'
        img = cv2.imread(img_full_path, cv2.IMREAD_COLOR)

        # already resized
        if np.shape(img) == (img_width, img_height, 3):
            continue

        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)  # resize with ANTI-ALIAS
        img = 5.0 * img - 4.0 * 255.0
        img = np.clip(img, 0.0, 255.0)
        cv2.imwrite(img_full_path, img)
