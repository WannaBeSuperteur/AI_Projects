
from generate_dataset.cnn_common import (load_dataset,
                                         load_remaining_images_dataset,
                                         load_cnn_model,
                                         train_cnn_models,
                                         predict_score_remaining_images)

import torch.nn as nn
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
INFERENCE_RESULT_DIR = f'{PROJECT_DIR_PATH}/stylegan/generate_dataset/cnn_inference_result'
IMAGE_RESOLUTION = 256


class QualityCNN(nn.Module):
    def __init__(self):
        super(QualityCNN, self).__init__()

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

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 12 * 4, 512),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x[:, :, : IMAGE_RESOLUTION // 4, IMAGE_RESOLUTION // 4 : 3 * IMAGE_RESOLUTION // 4]

        # Conv
        x = self.conv1(x)  # 126 x 62
        x = self.conv2(x)  # 124 x 60
        x = self.pool1(x)  # 62 x 30

        x = self.conv3(x)  # 60 x 28
        x = self.pool2(x)  # 30 x 14

        x = self.conv4(x)  # 28 x 12
        x = self.pool3(x)  # 14 x 6

        x = self.conv5(x)  # 12 x 4

        x = x.view(-1, 256 * 12 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# labeling 이 안 된 13,000 장에 대해 예측된 Quality 속성 값 반환
# Create Date : 2025.05.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - final_score (Pandas DataFrame) : Quality 속성 값에 대한 모델 예측값을 저장한 Pandas DataFrame
#                                    columns = ['img_no', 'img_path', 'property_quality_final_score',
#                                               'score_model_0', 'score_model_1', ...]

def main_quality():

    # load dataset
    data_loader = load_dataset(property_name='quality')

    # load or train model
    try:
        print('loading CNN models ...')
        cnn_models = load_cnn_model(property_name='quality', cnn_model_class=QualityCNN)
        print('loading CNN models successful!')

    except Exception as e:
        print(f'CNN model load failed : {e}')

        cnn_models = train_cnn_models(data_loader,
                                      is_stratified=True,
                                      property_name='quality',
                                      cnn_model_class=QualityCNN)

    # run inference on remaining 13,000 images
    remaining_image_loader = load_remaining_images_dataset(property_name='quality')
    report_path = f'{INFERENCE_RESULT_DIR}/quality.csv'
    os.makedirs(INFERENCE_RESULT_DIR, exist_ok=True)

    final_score = predict_score_remaining_images(property_name='quality',
                                                 remaining_images_loader=remaining_image_loader,
                                                 cnn_models=cnn_models,
                                                 report_path=report_path)

    print('FINAL PREDICTION SCORE (QUALITY) :\n')
    print(final_score)


if __name__ == '__main__':
    main_quality()
