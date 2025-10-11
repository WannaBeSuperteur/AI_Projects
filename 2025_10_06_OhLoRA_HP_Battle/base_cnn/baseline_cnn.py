
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

import numpy as np
import pandas as pd

import os
import sys


RESNET_LAST_LAYER_FEATURES = 1000
NUM_CLASSES = 10

torch.set_printoptions(linewidth=160, sci_mode=False)
np.set_printoptions(suppress=True)

default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


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


class ClassificationDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_path = dataset_df['img_path'].tolist()
        self.label = dataset_df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        label = self.label[idx]

        img = read_image(img_path)
        img = self.transform(img)

        return img, label


# PyTorch 데이터셋 생성
# Create Date : 2025.10.11
# Last Update Date : -

# args :
# - dataset_name (str) : 데이터셋 이름 ('mnist', 'fashion_mnist' or 'cifar_10')

# returns :
# - train_dataset (Dataset) : train, valid, 데이터로 분리할 학습 데이터셋
# - test_dataset  (Dataset) : 테스트 데이터셋

def create_pytorch_dataset(dataset_name):
    class_names = os.listdir(f'../datasets/{dataset_name}/train')
    class_names = sorted(list(set(class_names)))

    # get image info
    train_dataset_dict = {'img_path': [], 'label': []}
    test_dataset_dict = {'img_path': [], 'label': []}

    for name in class_names:
        train_img_dir = f'../datasets/{dataset_name}/train/{name}'
        test_img_dir = f'../datasets/{dataset_name}/test/{name}'
        train_img_names = os.listdir(train_img_dir)
        test_img_names = os.listdir(test_img_dir)
        train_img_paths = [os.path.join(train_img_dir, name) for name in train_img_names]
        test_img_paths = [os.path.join(test_img_dir, name) for name in test_img_names]

        train_dataset_dict['img_path'] += train_img_paths
        train_dataset_dict['label'] += [name for _ in range(len(train_img_paths))]
        test_dataset_dict['img_path'] += test_img_paths
        test_dataset_dict['label'] += [name for _ in range(len(test_img_paths))]

    # create Pandas DataFrame
    train_dataset_df = pd.DataFrame(train_dataset_dict)
    test_dataset_df = pd.DataFrame(test_dataset_dict)

    train_dataset_df.to_csv(f'../datasets/{dataset_name}_train.csv', index=False)
    test_dataset_df.to_csv(f'../datasets/{dataset_name}_test.csv', index=False)

    # return PyTorch dataset
    train_dataset = ClassificationDataset(train_dataset_df, default_transform)
    test_dataset = ClassificationDataset(test_dataset_df, default_transform)

    return train_dataset, test_dataset


if __name__ == '__main__':

    # add global common path
    global_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    sys.path.append(global_path)
    from global_common.torch_training import split_tv_and_t

    # load dataset
    dataset_names = ['mnist', 'fashion_mnist', 'cifar_10']

    for dataset_name in dataset_names:
        train_dataset, test_dataset = create_pytorch_dataset(dataset_name)
        train_loader, valid_loader, test_loader = split_tv_and_t(train_dataset, test_dataset, train_ratio=0.9)

        print(f'\ndataset name : {dataset_name}')
        print(f'train size : {len(train_loader.dataset)}')
        print(f'valid size : {len(valid_loader.dataset)}')
        print(f'test  size : {len(test_loader.dataset)}')
