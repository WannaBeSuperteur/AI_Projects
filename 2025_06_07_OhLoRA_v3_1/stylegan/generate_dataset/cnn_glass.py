
import torch.nn as nn
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
INFERENCE_RESULT_DIR = f'{PROJECT_DIR_PATH}/stylegan/generate_dataset/cnn_inference_result'
IMAGE_RESOLUTION = 256


class GlassCNN(nn.Module):
    def __init__(self):
        super(GlassCNN, self).__init__()

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
            nn.Linear(256 * 12 * 2, 512),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x[:, :,
              5 * IMAGE_RESOLUTION // 16 : IMAGE_RESOLUTION // 2,
              IMAGE_RESOLUTION // 4 : 3 * IMAGE_RESOLUTION // 4]

        # Conv
        x = self.conv1(x)  # 126 x 46
        x = self.conv2(x)  # 124 x 44
        x = self.pool1(x)  # 62 x 22

        x = self.conv3(x)  # 60 x 20
        x = self.pool2(x)  # 30 x 10

        x = self.conv4(x)  # 28 x 8
        x = self.pool3(x)  # 14 x 4

        x = self.conv5(x)  # 12 x 2

        x = x.view(-1, 256 * 12 * 2)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x

