
import torch
import torch.nn as nn


class AutoEncoderEncoder_1_28_28(nn.Module):
    def __init__(self):
        super(AutoEncoderEncoder_1_28_28, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

    def forward(self, x):

        # Conv
        x = self.conv1(x)  # 14
        x = self.conv2(x)  # 7
        x = self.conv3(x)  # 5
        x = self.conv4(x)  # 3

        return x


class AutoEncoderDecoder_1_28_28(nn.Module):
    def __init__(self):
        super(AutoEncoderDecoder_1_28_28, self).__init__()

        # DeConv Layers
        self.decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU()
        )
        self.decoder_deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU()
        )

    def forward(self, x):

        # DeConv Layers
        x = self.decoder_deconv1(x)  # 5
        x = self.decoder_deconv2(x)  # 7
        x = self.decoder_deconv3(x)  # 14
        x = self.decoder_deconv4(x)  # 28

        return x


class AutoEncoderEncoder_3_32_32(nn.Module):
    def __init__(self):
        super(AutoEncoderEncoder_3_32_32, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

    def forward(self, x):

        # Conv
        x = self.conv1(x)  # 16
        x = self.conv2(x)  # 8
        x = self.conv3(x)  # 6
        x = self.conv4(x)  # 4

        return x


class AutoEncoderDecoder_3_32_32(nn.Module):
    def __init__(self):
        super(AutoEncoderDecoder_3_32_32, self).__init__()

        # DeConv Layers
        self.decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU()
        )
        self.decoder_deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.LeakyReLU()
        )

    def forward(self, x):

        # DeConv Layers
        x = self.decoder_deconv1(x)  # 6
        x = self.decoder_deconv2(x)  # 8
        x = self.decoder_deconv3(x)  # 16
        x = self.decoder_deconv4(x)  # 32

        return x
