
import torch.nn as nn


# for 'fashion_mnist' and 'mnist' dataset
class BaseCNN_1_28_28(nn.Module):
    def get_conv_activation_func(self, activation_func):
        if activation_func == 'leaky_relu':
            return nn.LeakyReLU
        elif activation_func == 'relu':
            return nn.ReLU

    def __init__(self, dropout_conv_earlier, dropout_conv_later, dropout_fc, activation_func):
        super(BaseCNN_1_28_28, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.Tanh(),
            nn.Dropout(dropout_fc)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  # 26 x 26
        x = self.conv2(x)  # 24 x 24
        x = self.pool1(x)  # 12 x 12

        x = self.conv3(x)  # 10 x 10
        x = self.conv4(x)  # 8 x 8
        x = self.conv5(x)  # 6 x 6

        x = x.view(-1, 128 * 6 * 6)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# for 'cifar-10' dataset
class BaseCNN_3_32_32(nn.Module):
    def get_conv_activation_func(self, activation_func):
        if activation_func == 'leaky_relu':
            return nn.LeakyReLU
        elif activation_func == 'relu':
            return nn.ReLU

    def __init__(self, dropout_conv_earlier, dropout_conv_later, dropout_fc, activation_func):
        super(BaseCNN_3_32_32, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_earlier)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            self.get_conv_activation_func(activation_func)(),
            nn.Dropout2d(dropout_conv_later)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.Tanh(),
            nn.Dropout(dropout_fc)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  # 30 x 30
        x = self.conv2(x)  # 28 x 28
        x = self.pool1(x)  # 14 x 14

        x = self.conv3(x)  # 12 x 12
        x = self.pool2(x)  # 6 x 6

        x = self.conv4(x)  # 4 x 4

        x = x.view(-1, 128 * 4 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x

