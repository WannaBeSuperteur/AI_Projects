
import torch
import torch.nn as nn
import numpy as np


RESNET_LAST_LAYER_FEATURES = 1000
NUM_CLASSES = 10


torch.set_printoptions(linewidth=160, sci_mode=False)
np.set_printoptions(suppress=True)


class ResNetClassificationModel(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetClassificationModel, self).__init__()
        self.resnet_model = resnet_model
        self.num_classes = NUM_CLASSES
        self.final_linear = nn.Linear(RESNET_LAST_LAYER_FEATURES, self.num_classes)
        self.final_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet_model(x)
        x = self.final_linear(x)
        x = self.final_softmax(x)
        return x




