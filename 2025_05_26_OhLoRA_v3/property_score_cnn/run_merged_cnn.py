
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

import pandas as pd
import os

from run_train_cnn import HairstyleScoreCNN, get_dataloader
from common import save_model_structure_pdf

IMG_RESOLUTION = 256
EXAMPLE_BATCH_SIZE = 30
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
VALID_BATCH_SIZE = 4

EXISTING_PROPERTY_SCORE_CNN_PATH = f'{PROJECT_DIR_PATH}/property_score_cnn/models/stylegan_gen_fine_tuned_v2_cnn.pth'
HAIRSTYLE_SCORE_CNN_PATH = f'{PROJECT_DIR_PATH}/property_score_cnn/models/ohlora_v3_hairstyle_property_cnn.pth'
MERGED_SCORE_CNN_PATH = f'{PROJECT_DIR_PATH}/property_score_cnn/models/ohlora_v3_merged_property_cnn.pth'
TRAIN_LOG_DIR_PATH = f'{PROJECT_DIR_PATH}/property_score_cnn/train_logs'


# Eyes Score part of CNN
class EyesScoreCNN(nn.Module):
    def __init__(self):
        super(EyesScoreCNN, self).__init__()

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
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 14 * 4, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Linear(256, 1)

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 126 x 46
        x = self.conv2(x)  # 124 x 44
        x = self.pool1(x)  # 62 x 22

        x = self.conv3(x)  # 60 x 20
        x = self.pool2(x)  # 30 x 10

        x = self.conv4(x)  # 28 x 8
        x = self.pool3(x)  # 14 x 4

        x = x.view(-1, 128 * 14 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# Mouth Score part of CNN
class MouthScoreCNN(nn.Module):
    def __init__(self):
        super(MouthScoreCNN, self).__init__()

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
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6 * 4, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Linear(256, 1)

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 62 x 46
        x = self.conv2(x)  # 60 x 44
        x = self.pool1(x)  # 30 x 22

        x = self.conv3(x)  # 28 x 20
        x = self.pool2(x)  # 14 x 10

        x = self.conv4(x)  # 12 x 8
        x = self.pool3(x)  # 6 x 4

        x = x.view(-1, 128 * 6 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# Pose Score part of CNN
class PoseScoreCNN(nn.Module):
    def __init__(self):
        super(PoseScoreCNN, self).__init__()

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
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 14 * 6, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Linear(256, 1)

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 78 x 46
        x = self.conv2(x)  # 76 x 44
        x = self.pool1(x)  # 38 x 22

        x = self.conv3(x)  # 36 x 20
        x = self.pool2(x)  # 18 x 10

        x = self.conv4(x)  # 16 x 8
        x = self.conv5(x)  # 14 x 6

        x = x.view(-1, 128 * 14 * 6)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# Hair Color Score part of CNN
class HairColorScoreCNN(nn.Module):
    def __init__(self):
        super(HairColorScoreCNN, self).__init__()

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
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Linear(256, 1)

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 254 x 254
        x = self.conv2(x)  # 252 x 252
        x = self.pool1(x)  # 126 x 126

        x = self.conv3(x)  # 124 x 124
        x = self.pool2(x)  # 62 x 62

        x = self.conv4(x)  # 60 x 60
        x = self.pool3(x)  # 30 x 30

        x = self.conv5(x)  # 28 x 28
        x = self.pool4(x)  # 14 x 14

        x = self.conv6(x)  # 12 x 12
        x = self.pool5(x)  # 6 x 6

        x = x.view(-1, 128 * 6 * 6)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# Hair Length Score part of CNN
class HairLengthScoreCNN(nn.Module):
    def __init__(self):
        super(HairLengthScoreCNN, self).__init__()

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
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 12 * 4, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Linear(256, 1)

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 254 x 126
        x = self.conv2(x)  # 252 x 124
        x = self.pool1(x)  # 126 x 62

        x = self.conv3(x)  # 124 x 60
        x = self.pool2(x)  # 62 x 30

        x = self.conv4(x)  # 60 x 28
        x = self.pool3(x)  # 30 x 14

        x = self.conv5(x)  # 28 x 12
        x = self.pool4(x)  # 14 x 6

        x = self.conv6(x)  # 12 x 4

        x = x.view(-1, 128 * 12 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# Background Mean & Std Score part of CNN
class BackgroundMeanStdScoreCNN(nn.Module):
    def __init__(self):
        super(BackgroundMeanStdScoreCNN, self).__init__()

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
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 12 * 4, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Linear(256, 2)

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 254 x 126
        x = self.conv2(x)  # 252 x 124
        x = self.pool1(x)  # 126 x 62

        x = self.conv3(x)  # 124 x 60
        x = self.pool2(x)  # 62 x 30

        x = self.conv4(x)  # 60 x 28
        x = self.pool3(x)  # 30 x 14

        x = self.conv5(x)  # 28 x 12
        x = self.pool4(x)  # 14 x 6

        x = self.conv6(x)  # 12 x 4

        x = x.view(-1, 128 * 12 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


class MergedPropertyScoreCNN(nn.Module):
    def __init__(self):
        super(MergedPropertyScoreCNN, self).__init__()

        # Conv Layers
        self.eyes_score_cnn = EyesScoreCNN()
        self.mouth_score_cnn = MouthScoreCNN()
        self.pose_score_cnn = PoseScoreCNN()
        self.hair_color_score_cnn = HairColorScoreCNN()
        self.hair_length_score_cnn = HairLengthScoreCNN()
        self.background_score_cnn = BackgroundMeanStdScoreCNN()
        self.hairstyle_score_cnn = HairstyleScoreCNN()

    def forward(self, x):
        x_eyes        = x[:, :,                                                   # for eyes score
                          3 * IMG_RESOLUTION // 8 : 9 * IMG_RESOLUTION // 16,
                          IMG_RESOLUTION // 4 : 3 * IMG_RESOLUTION // 4]
        x_entire      = x                                                         # for hair color score
        x_bottom_half = x[:, :, IMG_RESOLUTION // 2:, :]                          # for hair length score
        x_mouth       = x[:, :,                                                   # for mouth score
                          5 * IMG_RESOLUTION // 8 : 13 * IMG_RESOLUTION // 16,
                          3 * IMG_RESOLUTION // 8 : 5 * IMG_RESOLUTION // 8]
        x_pose        = x[:, :,                                                   # for pose score
                          7 * IMG_RESOLUTION // 16 : 5 * IMG_RESOLUTION // 8,
                          11 * IMG_RESOLUTION // 32 : 21 * IMG_RESOLUTION // 32]
        x_upper_half  = x[:, :, : IMG_RESOLUTION // 2, :]                         # for background mean, std score

        # Compute Each Score
        x_eyes = self.eyes_score_cnn(x_eyes)
        x_entire = self.hair_color_score_cnn(x_entire)
        x_bottom_half = self.hair_length_score_cnn(x_bottom_half)
        x_mouth = self.mouth_score_cnn(x_mouth)
        x_pose = self.pose_score_cnn(x_pose)
        x_upper_half = self.background_score_cnn(x_upper_half)
        x_hairstyle = self.hairstyle_score_cnn(x)

        # Final Concatenate (SAME as all_scores_v2.csv column order :
        #                    eyes, hair-color, hair-length, mouth, pose, back-mean, back-std, hairstyle)
        x_final = torch.concat([x_eyes, x_entire, x_bottom_half, x_mouth, x_pose, x_upper_half, x_hairstyle],
                               dim=1)

        return x_final


# Inference Test 실시
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - merged_property_cnn_model (nn.Module) : Pre-trained weight 이 로딩된 Merged Property Score CNN 모델
# - before_after              (str)       : weight load 이전/이후 여부를 나타내는 값

# File Outputs:
# - train_logs/merged_cnn_inference_test_before_weight_load.csv : weight load 이전 inference test 결과
# - train_logs/merged_cnn_inference_test_after_weight_load.csv  : weight load 이후 inference test 결과

def run_inference_test(cnn_model, before_after):

    # load dataloader and split dataset
    hairstyle_score_dataloader = get_dataloader()

    dataset_size = len(hairstyle_score_dataloader.dataset)
    train_size = int(dataset_size * 0.99)
    valid_size = dataset_size - train_size

    _, valid_dataset = random_split(hairstyle_score_dataloader.dataset, [train_size, valid_size])
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    # prepare inference test result dict
    inference_test_result = {
        'img_path': [],
        'eyes_score': [], 'hair_color_score': [], 'hair_length_score': [], 'mouth_score': [],
        'pose_score': [], 'background_mean_score': [], 'background_std_score': [], 'hairstyle_score': []
    }

    # run inference test
    with torch.no_grad():
        for idx, raw_data in enumerate(valid_loader):
            image_paths = raw_data['img_path']
            images = raw_data['image'].to(cnn_model.device)
            outputs = cnn_model(images).to(torch.float32).detach().cpu().numpy()

            inference_test_result['img_path'] += image_paths

            current_batch_size = len(image_paths)
            for i in range(current_batch_size):
                inference_test_result['eyes_score'].append(round(outputs[i][0], 4))
                inference_test_result['hair_color_score'].append(round(outputs[i][1], 4))
                inference_test_result['hair_length_score'].append(round(outputs[i][2], 4))
                inference_test_result['mouth_score'].append(round(outputs[i][3], 4))
                inference_test_result['pose_score'].append(round(outputs[i][4], 4))
                inference_test_result['background_mean_score'].append(round(outputs[i][5], 4))
                inference_test_result['background_std_score'].append(round(outputs[i][6], 4))
                inference_test_result['hairstyle_score'].append(round(outputs[i][7], 4))

    # save inference test result
    inference_test_result_path = f'{TRAIN_LOG_DIR_PATH}/merged_cnn_inference_test_{before_after}_weight_load.csv'
    inference_test_df = pd.DataFrame(inference_test_result)
    inference_test_df.to_csv(inference_test_result_path)


# Pre-trained weight 이 로딩된 Merged Property Score CNN 모델 불러오기
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - device         (device) : CNN 모델을 mapping 시킬 device (GPU 등)
# - inference_test (bool)   : state dict 로딩 전후 inference test 실시 여부

# Returns:
# - merged_property_cnn_model (nn.Module) : Pre-trained weight 이 로딩된 Merged Property Score CNN 모델

def load_merged_property_cnn_model(device, inference_test=False):
    merged_property_cnn_model = MergedPropertyScoreCNN()

    existing_cnn_state_dict = torch.load(EXISTING_PROPERTY_SCORE_CNN_PATH, map_location=device, weights_only=False)
    hairstyle_cnn_state_dict = torch.load(HAIRSTYLE_SCORE_CNN_PATH, map_location=device, weights_only=False)

    save_model_structure_pdf(model=merged_property_cnn_model,
                             model_name='merged_property_cnn',
                             input_size=(EXAMPLE_BATCH_SIZE, 3, IMG_RESOLUTION, IMG_RESOLUTION))

    # device mapping
    merged_property_cnn_model.to(device)
    merged_property_cnn_model.device = device

    # inference test before loading state dict
    if inference_test:
        run_inference_test(merged_property_cnn_model, before_after='before')

    # load state dict
    merged_property_cnn_model.load_state_dict(existing_cnn_state_dict, strict=False)
    merged_property_cnn_model.load_state_dict(hairstyle_cnn_state_dict, strict=False)

    # inference test after loading state dict
    if inference_test:
        run_inference_test(merged_property_cnn_model, before_after='after')

    return merged_property_cnn_model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for merged CNN loading : {device}')

    merged_property_cnn_model = load_merged_property_cnn_model(device, inference_test=True)
    torch.save(merged_property_cnn_model.state_dict(), MERGED_SCORE_CNN_PATH)
