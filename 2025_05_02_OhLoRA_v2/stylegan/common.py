
import torch
from torchinfo import summary
from torchview import draw_graph

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan/model_structure_pdf'


TRAIN_BATCH_SIZE = 16
IMAGE_RESOLUTION = 256

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])

stylegan_transform_for_augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# Image Dataset with Property Scores
class PropertyScoreImageDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_paths = dataset_df['img_path'].tolist()
        self.transform = transform

        self.eyes_scores = dataset_df['eyes_score'].tolist()
        self.mouth_scores = dataset_df['mouth_score'].tolist()
        self.pose_scores = dataset_df['pose_score'].tolist()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)

        eyes_score = self.eyes_scores[idx]
        mouth_score = self.mouth_scores[idx]
        pose_score = self.pose_scores[idx]

        property_scores = {'eyes': eyes_score,
                           'mouth': mouth_score,
                           'pose': pose_score}

        # normalize
        image = self.transform(image)

        # return simplified image path
        simplified_img_path = '/'.join(img_path.split('/')[-2:])

        return {'image': image, 'label': property_scores, 'img_path': simplified_img_path}


# StyleGAN Fine-Tuning 용 데이터셋의 Data Loader 로딩
# Create Date : 2025.05.03
# Last Update Date : 2025.05.04
# - Transformation (image augmentation) 에 brightness, contrast 추가 반영

# Arguments:
# - 없음

# Returns:
# - stylegan_ft_loader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

def get_stylegan_fine_tuning_dataloader():
    property_score_csv_path = f'{PROJECT_DIR_PATH}/stylegan/all_scores_v2_cnn.csv'
    property_score_df = pd.read_csv(property_score_csv_path)

    stylegan_ft_dataset = PropertyScoreImageDataset(dataset_df=property_score_df,
                                                    transform=stylegan_transform_for_augmentation)

    stylegan_ft_loader = DataLoader(stylegan_ft_dataset,
                                    batch_size=TRAIN_BATCH_SIZE,
                                    shuffle=True)

    return stylegan_ft_loader


# Model Summary (모델 구조) 출력
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - model               (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator 또는 Discriminator
# - model_name          (str)       : 모델을 나타내는 이름
# - input_size          (tuple)     : 모델에 입력될 데이터의 입력 크기
# - print_layer_details (bool)      : 각 레이어 별 detailed info 출력 여부
# - print_frozen        (bool)      : 각 레이어가 freeze 되었는지의 상태 출력 여부

def print_summary(model, model_name, input_size, print_layer_details=False, print_frozen=False):
    print(f'\n\n==== MODEL SUMMARY : {model_name} ====\n')
    summary(model, input_size=input_size)

    if print_layer_details:
        print(model)

    if print_frozen:
        for name, param in model.named_parameters():
            print(f'layer name = {name:40s}, trainable = {param.requires_grad}')


# 기존 Pre-train 된 StyleGAN 모델의 구조를 PDF 로 내보내기
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - model               (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator 또는 Discriminator
# - model_name          (str)       : 모델을 나타내는 이름
# - input_size          (tuple)     : 모델에 입력될 데이터의 입력 크기
# - print_layer_details (bool)      : 각 레이어 별 detailed info 출력 여부
# - print_frozen        (bool)      : 각 레이어가 freeze 되었는지의 상태 출력 여부

def save_model_structure_pdf(model, model_name, input_size, print_layer_details=False, print_frozen=False):
    model_graph = draw_graph(model, input_size=input_size, depth=5)
    visual_graph = model_graph.visual_graph

    # Model Graph 이미지 저장
    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/{model_name}.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    # Model Summary 출력
    print_summary(model, model_name, input_size, print_layer_details=print_layer_details, print_frozen=print_frozen)


# 기존 Oh-LoRA Project 에서 Pre-train 된 StyleGAN (StyleGAN-FineTune-v1) 모델 로딩 (Generator 의 state dict 만)
# Create Date : 2025.05.03
# Last Update Date : 2025.05.04
# - 모델 디렉토리 이름 변경 (stylegan_models -> models) 반영

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict

def load_existing_stylegan_finetune_v1(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v1.pth'

    # load generator state dict
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)

    return generator_state_dict


# 기존 StyleGAN-VectorFind-v6 모델 로딩 (Generator 의 state dict 만)
# Create Date : 2025.05.09
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - generator_state_dict (OrderedDict) : StyleGAN-VectorFind-v6 모델의 Generator 의 state_dict

def load_existing_stylegan_vectorfind_v6(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_vector_find_v6.pth'

    # load generator state dict
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)

    return generator_state_dict
