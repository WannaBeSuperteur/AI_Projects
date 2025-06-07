
from torchinfo import summary
from torchview import draw_graph
import torchvision.transforms as transforms

import torch
import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)


MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan/model_structure_pdf'
MERGED_PROPERTY_SCORE_CNN_PATH = f'{PROJECT_DIR_PATH}/property_score_cnn/models/ohlora_v3_merged_property_cnn.pth'

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# Model Summary (모델 구조) 출력
# Create Date : 2025.05.26
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
# Create Date : 2025.05.26
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
    os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)
    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/{model_name}.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    # Model Summary 출력
    print_summary(model, model_name, input_size, print_layer_details=print_layer_details, print_frozen=print_frozen)


# 기존 Oh-LoRA v1 Project 에서 Pre-train 된 StyleGAN (StyleGAN-FineTune-v1) 모델 로딩 (Generator 의 state dict 만)
# Create Date : 2025.05.26
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


# 기존 Oh-LoRA v1 Project 에서 Pre-train 된 StyleGAN (StyleGAN-FineTune-v1) 모델 로딩 (Discriminator 까지 포함)
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - generator_state_dict     (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Discriminator 의 state_dict

def load_existing_stylegan_finetune_v1_all(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v1.pth'
    discriminator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_dis_fine_tuned_v1.pth'

    # load generator state dict
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    discriminator_state_dict = torch.load(discriminator_path, map_location=device, weights_only=True)

    return generator_state_dict, discriminator_state_dict


# StyleGAN-FineTune-v8 의 Generator 모델 로딩
# Create Date : 2025.05.29
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
# Create Date : 2025.05.29
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


