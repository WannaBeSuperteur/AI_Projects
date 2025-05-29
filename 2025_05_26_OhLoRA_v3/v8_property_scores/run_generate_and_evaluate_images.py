import torch


# StyleGAN-FineTune-v8 모델의 Generator 로딩
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - finetune_v8_generator (nn.Module) : StyleGAN-FineTune-v8 모델의 Generator

def load_stylegan_finetune_v8_generator(device):
    raise NotImplementedError


# Merged Property Score CNN (hairstyle 포함한 핵심 속성 값 계산용 CNN) 로딩
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - merged_property_score_cnn (nn.Module) : Merged Property Score CNN (핵심 속성 값 계산용 CNN)

def load_merged_property_score_cnn(device):
    raise NotImplementedError


# StyleGAN-FineTune-v8 모델의 Generator 를 이용하여 이미지 생성
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - finetune_v8_generator (nn.Module) : StyleGAN-FineTune-v8 모델의 Generator
# - n                     (int)       : 생성할 이미지 개수

def generate_images(finetune_v8_generator, n=5000):
    raise NotImplementedError


# 생성된 이미지에 대해 핵심 속성 값 (hair_color, hair_length, background_score, hairstyle) 도출 및 그 결과를 csv로 저장
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - merged_property_score_cnn (nn.Module) : Merged Property Score CNN (핵심 속성 값 계산용 CNN)

def compute_property_scores(merged_property_score_cnn):
    raise NotImplementedError


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for generating image using StyleGAN-FineTune-v8 : {device}')

    finetune_v8_generator = load_stylegan_finetune_v8_generator(device)
    merged_property_score_cnn = load_merged_property_score_cnn(device)

    generate_images(finetune_v8_generator)
    compute_property_scores(merged_property_score_cnn)
