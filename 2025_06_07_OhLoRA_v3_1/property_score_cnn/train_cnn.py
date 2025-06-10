import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMG_RESOLUTION = 256

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# Image Dataset with Property Scores
class PropertyScoreImageDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_paths = dataset_df['img_path'].tolist()
        self.transform = transform

        self.hairstyle_scores = dataset_df['hairstyle_score'].tolist()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)

        hairstyle_score = self.hairstyle_scores[idx]
        property_scores = {'hairstyle': hairstyle_score}

        # normalize
        image = self.transform(image)

        # return simplified image path
        simplified_img_path = '/'.join(img_path.split('/')[-2:])

        return {'image': image, 'label': property_scores, 'img_path': simplified_img_path}


# Hairstyle Score CNN (곱슬머리 vs. 직모)

class HairstyleScorePartCNN(nn.Module):
    def __init__(self):
        super(HairstyleScorePartCNN, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

    def forward(self, x):

        # Conv
        x = self.conv1(x)  # 94 x 126
        x = self.conv2(x)  # 92 x 124
        x = self.pool1(x)  # 46 x 62

        x = self.conv3(x)  # 46 x 60
        x = self.pool2(x)  # 22 x 30

        x = self.conv4(x)  # 20 x 28
        x = self.pool3(x)  # 10 x 14

        x = self.conv5(x)  # 8 x 12
        x = self.conv6(x)  # 6 x 10

        x = x.view(-1, 128 * 6 * 10)
        return x


class HairstyleScoreCNN(nn.Module):
    def __init__(self):
        super(HairstyleScoreCNN, self).__init__()

        # Conv Layers
        self.bottom_left_cnn = HairstyleScorePartCNN()
        self.bottom_right_cnn = HairstyleScorePartCNN()

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(2 * (128 * 6 * 10), 512),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.fc_final = nn.Linear(512, 1)

    def forward(self, x):
        x_bottom_left = x[:, :, IMG_RESOLUTION // 2:, :3 * IMG_RESOLUTION // 8]
        x_bottom_right = x[:, :, IMG_RESOLUTION // 2:, 5 * IMG_RESOLUTION // 8:]

        x_bottom_left = self.bottom_left_cnn(x_bottom_left)
        x_bottom_right = self.bottom_right_cnn(x_bottom_right)

        # Fully Connected
        final_x = torch.concat([x_bottom_left, x_bottom_right], dim=1)
        final_x = self.fc1(final_x)
        final_x = self.fc_final(final_x)

        return final_x


# 데이터셋 (CNN 에 의해 계산된 hairstyle property scores) 의 Data Loader 로딩
# Create Date : 2025.06.10

# Arguments:
# - 없음

# Returns:
# - hairstyle_score_dataloader (DataLoader) : hairstyle scores 데이터셋의 Data Loader

def get_dataloader():
    all_scores_dir_path = f'{PROJECT_DIR_PATH}/property_score_cnn/segmentation/property_score_results'
    property_score_csv_path = f'{all_scores_dir_path}/all_scores_ohlora_v3.csv'
    property_score_df = pd.read_csv(property_score_csv_path)

    dataset = PropertyScoreImageDataset(dataset_df=property_score_df, transform=stylegan_transform)
    hairstyle_score_dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    return hairstyle_score_dataloader

