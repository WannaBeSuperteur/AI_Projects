
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


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
    raise NotImplementedError


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
    raise NotImplementedError


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


if __name__ == '__main__':
    dataset_path = f'{PROJECT_DIR_PATH}/stylegan/generated_face_images_filtered'
    train_dataloader, valid_dataloader, test_dataloader = generate_dataset(dataset_path)
    model = define_segmentation_model()

    best_epoch_model = train_model(model, train_dataloader, valid_dataloader)
    test_result_dict = test_model(model, test_dataloader)

    model_save_path = f'{PROJECT_DIR_PATH}/segmenation/models/segmentation_model_ohlora_v4'
    torch.save(best_epoch_model.state_dict(), model_save_path)
