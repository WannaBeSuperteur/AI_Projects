import torch
import torch.nn as nn
import os


IMG_HEIGHT = 256
IMG_WIDTH = 256

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
INFERENCE_RESULT_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/inference_result'


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

        self.conv5 = nn.Sequential(
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
        self.fc_final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 126 x 62
        x = self.conv2(x)  # 124 x 60
        x = self.pool1(x)  # 62 x 30

        x = self.conv3(x)  # 60 x 28
        x = self.pool2(x)  # 30 x 14

        x = self.conv4(x)  # 28 x 12
        x = self.pool3(x)  # 14 x 6

        x = self.conv5(x)  # 12 x 4

        x = x.view(-1, 128 * 12 * 4)

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

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 10 * 6, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 126 x 94
        x = self.conv2(x)  # 124 x 92
        x = self.pool1(x)  # 62 x 46

        x = self.conv3(x)  # 60 x 44
        x = self.pool2(x)  # 30 x 22

        x = self.conv4(x)  # 28 x 20
        x = self.pool3(x)  # 14 x 10

        x = self.conv5(x)  # 12 x 8
        x = self.conv6(x)  # 10 x 6

        x = x.view(-1, 128 * 10 * 6)

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
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Conv
        x = self.conv1(x)  # 62 x 62
        x = self.conv2(x)  # 60 x 60
        x = self.pool1(x)  # 30 x 30

        x = self.conv3(x)  # 28 x 28
        x = self.pool2(x)  # 14 x 14

        x = self.conv4(x)  # 12 x 12
        x = self.pool3(x)  # 6 x 6

        x = x.view(-1, 128 * 6 * 6)

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
        self.fc_final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

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
        self.fc_final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

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
        self.fc_final = nn.Sequential(
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

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


class PropertyScoreCNN(nn.Module):
    def __init__(self):
        super(PropertyScoreCNN, self).__init__()

        # Conv Layers
        self.eyes_score_cnn = EyesScoreCNN()
        self.mouth_score_cnn = MouthScoreCNN()
        self.pose_score_cnn = PoseScoreCNN()
        self.hair_color_score_cnn = HairColorScoreCNN()
        self.hair_length_score_cnn = HairLengthScoreCNN()
        self.background_score_cnn = BackgroundMeanStdScoreCNN()

    def forward(self, x):
        x_eyes = x[:, :, IMG_HEIGHT // 4 : IMG_HEIGHT // 2, IMG_WIDTH // 4 : 3 * IMG_WIDTH // 4]
        x_mouth = x[:, :, IMG_HEIGHT // 2 : 7 * IMG_HEIGHT // 8, IMG_WIDTH // 4 : 3 * IMG_WIDTH // 4]
        x_nose = x[:, :, 3 * IMG_HEIGHT // 8 : 5 * IMG_HEIGHT // 8, 3 * IMG_WIDTH // 8 : 5 * IMG_WIDTH // 8]
        x_entire = x
        x_bottom_half = x[:, :, IMG_HEIGHT // 2 :, :]
        x_upper_half = x[:, :, : IMG_HEIGHT // 2, :]

        # Compute Each Score
        x_eyes = self.eyes_score_cnn(x_eyes)
        x_mouth = self.mouth_score_cnn(x_mouth)
        x_nose = self.pose_score_cnn(x_nose)
        x_entire = self.hair_color_score_cnn(x_entire)
        x_bottom_half = self.hair_length_score_cnn(x_bottom_half)
        x_upper_half = self.background_score_cnn(x_upper_half)

        # Final Concatenate
        x_final = torch.concat([x_eyes, x_mouth, x_nose, x_entire, x_bottom_half, x_upper_half], dim=1)
        return x_final


# CNN 모델 정의
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - device (device) : CNN 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - cnn_model (nn.Module) : 학습할 CNN 모델

def define_cnn_model(device):
    cnn_model = PropertyScoreCNN()
    cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.00005)
    cnn_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=cnn_model.optimizer,
                                                                     T_max=10,
                                                                     eta_min=0)

    cnn_model.to(device)
    cnn_model.device = device

    return cnn_model


# CNN 모델 학습
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - device                 (device)     : CNN 모델을 mapping 시킬 device (GPU 등)
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - cnn_model (nn.Module) : 학습된 CNN 모델

def train_cnn_model(device, fine_tuning_dataloader):
    cnn_model = define_cnn_model(device)

    raise NotImplementedError


# CNN 모델의 Train Step
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - cnn_model              (nn.Module)  : 학습 중인 CNN 모델
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

def train_cnn_train_step(cnn_model, fine_tuning_dataloader):
    raise NotImplementedError


# CNN 모델의 Valid Step
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - cnn_model              (nn.Module)  : 학습 중인 CNN 모델
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - valid_log (dict) : CNN 모델의 Validation Log
#                      {'epoch': int, 'valid_loss': float, 'eyes_score_loss': float, 'mouth_score_loss': float,
#                       'pose_score_loss': float, 'hair_color_score_loss': float, 'hair_length_score_loss': float,
#                       'back_mean_score_loss': float, 'back_std_score_loss': float}

def train_cnn_valid_step(cnn_model, fine_tuning_dataloader):
    raise NotImplementedError


# 학습된 CNN 모델 불러오기
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - cnn_model_path (str)    : CNN 모델 저장 경로
# - device         (device) : CNN 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - cnn_model (nn.Module) : 학습된 CNN 모델

def load_cnn_model(cnn_model_path, device):
    cnn_model = PropertyScoreCNN()
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device, weights_only=True))

    cnn_model.to(device)
    cnn_model.device = device

    return cnn_model


# StyleGAN-FineTune-v2 모델 Fine-Tuning 실시
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - generator              (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator     (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator
# - fine_tuned_generator_cnn (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Discriminator

def run_fine_tuning(generator, fine_tuning_dataloader):
    cnn_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v2.pth'

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v2 : {device}')

    # load CNN model
    try:
        cnn_model = load_cnn_model(cnn_model_path, device)

    except Exception as e:
        print(f'cnn model load failed : {e}')

        # train CNN model
        cnn_model = train_cnn_model(device, fine_tuning_dataloader)
        torch.save(cnn_model.state_dict(), cnn_model_path)

    raise NotImplementedError
