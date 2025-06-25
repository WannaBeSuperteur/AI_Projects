
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from seg_model_ohlora_v4.model import SegModelForOhLoRAV4

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4


class OhLoRAV4SegmentationModelDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx])

        return image, mask


# Oh-LoRA v4 용 경량화된 Segmentation Model 정의
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - model (nn.Module) : 경량화 모델

def define_segmentation_model():
    model = SegModelForOhLoRAV4()
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model.optimizer, gamma=0.975)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.device = device

    return model


# Oh-LoRA v4 용 경량화된 Segmentation Model 의 데이터셋 생성
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - dataset_path (str) : image 와 mask 가 저장되어 있는 데이터셋 경로

# Returns:
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader
# - test_dataloader  (DataLoader) : 경량화 모델의 Test Data Loader

def generate_dataset(dataset_path):
    image_paths = list(filter(lambda x: x.startswith('face'), os.listdir(dataset_path)))
    mask_paths = list(filter(lambda x: x.startswith('parsing_hair_label'), os.listdir(dataset_path)))

    image_paths = [f'{dataset_path}/{image_path}' for image_path in image_paths]
    mask_paths = [f'{dataset_path}/{mask_path}' for mask_path in mask_paths]

    # define dataset
    dataset = OhLoRAV4SegmentationModelDataset(image_paths, mask_paths)

    # define dataloader
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = int(dataset_size * 0.1)
    test_size = dataset_size - (train_size + valid_size)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader

# Returns:
# - best_epoch_model (nn.Module) : best epoch (Loss 최소) 에서의 Segmentation Model

def train_model(model, train_dataloader, valid_dataloader):
    raise NotImplementedError


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습 (각 epoch)
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader

def train_model_each_epoch(model, train_dataloader, valid_dataloader):
    raise NotImplementedError


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습 (PyTorch Train step)
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader

def run_train_step(model, train_dataloader):
    raise NotImplementedError


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습 (PyTorch Valid step)
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader

def run_valid_step(model, valid_dataloader):
    raise NotImplementedError


# Oh-LoRA v4 용 경량화된 Segmentation Model 성능 테스트
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model           (nn.Module)  : 학습된 경량화 모델
# - test_dataloader (DataLoader) : 경량화 모델의 Test Data Loader

# Returns:
# - test_result_dict (dict) : 테스트 결과 성능지표 dict
#                             {'mse': float, 'iou': float, 'dice': float, 'recall': float, 'precision': float}

def test_model(model, test_dataloader):
    raise NotImplementedError


def main():
    dataset_path = f'{PROJECT_DIR_PATH}/segmentation/segmentation_results_facexformer'
    train_dataloader, valid_dataloader, test_dataloader = generate_dataset(dataset_path)
    model = define_segmentation_model()

    best_epoch_model = train_model(model, train_dataloader, valid_dataloader)
    test_result_dict = test_model(model, test_dataloader)

    model_save_path = f'{PROJECT_DIR_PATH}/segmenation/models/segmentation_model_ohlora_v4'
    torch.save(best_epoch_model.state_dict(), model_save_path)
