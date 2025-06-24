import torch
import torch.nn as nn

IMG_RESOLUTION = 256


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
