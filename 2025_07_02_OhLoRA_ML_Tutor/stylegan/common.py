
import torch
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# 기존 StyleGAN-VectorFind-v8 모델 로딩 (Generator 의 state dict 만)
# Create Date : 2025.08.01
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
