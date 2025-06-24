import torchvision.transforms as transforms
import torch
import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from stylegan.merged_property_score_cnn import MergedPropertyScoreCNN


MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan/model_structure_pdf'
MERGED_PROPERTY_SCORE_CNN_PATH = f'{PROJECT_DIR_PATH}/stylegan/models/ohlora_v3_merged_property_cnn.pth'

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# 기존 Oh-LoRA Project 에서 Pre-train 된 StyleGAN (StyleGAN-FineTune-v1) 모델 로딩 (Generator 의 state dict 만)
# Create Date : 2025.06.24
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict

def load_existing_stylegan_finetune_v1(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v1.pth'

    # load generator state dict
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    return generator_state_dict


# 기존 StyleGAN-VectorFind-v7 모델 로딩 (Generator 의 state dict 만)
# Create Date : 2025.06.24
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - generator_state_dict (OrderedDict) : StyleGAN-VectorFind-v6 모델의 Generator 의 state_dict

def load_existing_stylegan_vectorfind_v7(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_vector_find_v7.pth'

    # load generator state dict
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    return generator_state_dict


# StyleGAN-FineTune-v8 의 Generator 모델 로딩
# Create Date : 2025.06.24
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v8 모델의 Generator 의 state_dict

def load_existing_stylegan_finetune_v8(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v8.pth'

    # load generator state dict
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    return generator_state_dict


# 기존 StyleGAN-VectorFind-v8 모델 로딩 (Generator 의 state dict 만)
# Create Date : 2025.06.24
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - generator_state_dict (OrderedDict) : StyleGAN-VectorFind-v6 모델의 Generator 의 state_dict

def load_existing_stylegan_vectorfind_v8(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_vector_find_v8.pth'

    # load generator state dict
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    return generator_state_dict


# Merged Property Score CNN (hairstyle 포함한 핵심 속성 값 계산용 CNN) 로딩
# Create Date : 2025.06.24
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - merged_property_score_cnn (nn.Module) : Merged Property Score CNN (핵심 속성 값 계산용 CNN)

def load_merged_property_score_cnn(device):
    merged_property_cnn_model = MergedPropertyScoreCNN()
    merged_property_cnn_state_dict = torch.load(MERGED_PROPERTY_SCORE_CNN_PATH,
                                                map_location=device,
                                                weights_only=False)
    merged_property_cnn_model.load_state_dict(merged_property_cnn_state_dict)

    merged_property_cnn_model.to(device)
    merged_property_cnn_model.device = device

    return merged_property_cnn_model
