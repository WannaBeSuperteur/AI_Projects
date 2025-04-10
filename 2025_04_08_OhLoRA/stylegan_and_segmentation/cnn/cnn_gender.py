
from cnn.cnn_common import (load_dataset,
                            load_remaining_images_dataset,
                            load_cnn_model,
                            train_cnn_models,
                            predict_score_remaining_images)

import torch
import torch.nn as nn

import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
INFERENCE_RESULT_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/inference_result'


class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()

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
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Conv
        x = self.conv1(x)  # 254
        x = self.conv2(x)  # 252
        x = self.pool1(x)  # 126

        x = self.conv3(x)  # 124
        x = self.pool2(x)  # 62

        x = self.conv4(x)  # 60
        x = self.pool3(x)  # 30

        x = self.conv5(x)  # 28
        x = self.pool4(x)  # 14

        x = self.conv6(x)  # 12
        x = self.pool5(x)  # 6

        x = x.view(-1, 256 * 6 * 6)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# labeling 이 안 된 8,000 장에 대해 예측된 Gender 속성 값 반환
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - final_score (Pandas DataFrame) : Gender 속성 값에 대한 모델 예측값을 저장한 Pandas DataFrame
#                                    columns = ['img_no', 'img_path', 'property_gender_final_score',
#                                               'score_model_0', 'score_model_1', ...]

def main_gender():

    # load dataset
    data_loader = load_dataset(property_name='gender')

    # load or train model
    try:
        print('loading CNN models ...')
        cnn_models = load_cnn_model(property_name='gender', cnn_model_class=GenderCNN)
        print('loading CNN models successful!')

    except Exception as e:
        print(f'CNN model load failed : {e}')
        cnn_models = train_cnn_models(data_loader)

    # run inference on remaining 8,000 images
    remaining_image_loader = load_remaining_images_dataset()
    report_path = f'{INFERENCE_RESULT_DIR}/gender.csv'
    final_score = predict_score_remaining_images(remaining_image_loader, cnn_models, report_path)

    print('FINAL PREDICTION SCORE (GENDER) :\n')
    print(final_score)


if __name__ == '__main__':
    main_gender()
