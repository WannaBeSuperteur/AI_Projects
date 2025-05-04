import torch
import torch.nn as nn


IMG_RESOLUTION = 256


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


class DiscriminatorForV5(nn.Module):
    def __init__(self):
        super(DiscriminatorForV5, self).__init__()

        # Conv Layers
        self.eyes_score_cnn = EyesScoreCNN()
        self.mouth_score_cnn = MouthScoreCNN()
        self.pose_score_cnn = PoseScoreCNN()


    def forward(self, x, label):
        x_eyes = x[:, :,                                                    # for eyes score
                   3 * IMG_RESOLUTION // 8 : 9 * IMG_RESOLUTION // 16,
                   IMG_RESOLUTION // 4 : 3 * IMG_RESOLUTION // 4]

        x_mouth = x[:, :,                                                   # for mouth score
                    5 * IMG_RESOLUTION // 8 : 13 * IMG_RESOLUTION // 16,
                    3 * IMG_RESOLUTION // 8 : 5 * IMG_RESOLUTION // 8]

        x_pose = x[:, :,                                                    # for pose score
                   7 * IMG_RESOLUTION // 16 : 5 * IMG_RESOLUTION // 8,
                   11 * IMG_RESOLUTION // 32 : 21 * IMG_RESOLUTION // 32]

        # Compute Each Score
        x_eyes_score = self.eyes_score_cnn(x_eyes)
        x_mouth_score = self.mouth_score_cnn(x_mouth)
        x_pose_score = self.pose_score_cnn(x_pose)

        # Final Concatenate
        x_concat = torch.concat([x_eyes_score, x_mouth_score, x_pose_score, label], dim=1)
        return x_concat
