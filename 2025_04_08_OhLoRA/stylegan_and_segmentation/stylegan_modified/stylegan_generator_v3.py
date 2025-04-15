import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np

from stylegan_modified.stylegan_generator import StyleGANGeneratorForV2
from stylegan_modified.stylegan_generator_v2 import load_cnn_model
from stylegan_modified.stylegan_generator_v3_gen_model import train_stylegan_finetune_v3

torch.set_printoptions(linewidth=160, sci_mode=False)
np.set_printoptions(suppress=True)


IMG_RESOLUTION = 256
PROPERTY_DIMS_Z = 7  # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator 불러오기
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - v3_gen_path (str)    : StyleGAN-FineTune-v3 모델 저장 경로
# - device      (device) : StyleGAN-FineTune-v3 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator

def load_stylegan_finetune_v3_model(v3_gen_path, device):
    fine_tuned_generator = StyleGANGeneratorForV2(resolution=IMG_RESOLUTION)  # Style-Mixing 미 적용된 v2 model 그대로 사용
    fine_tuned_generator.load_state_dict(torch.load(v3_gen_path, map_location=device, weights_only=False))

    fine_tuned_generator.to(device)
    fine_tuned_generator.device = device

    return fine_tuned_generator


# StyleGAN-FineTune-v3 모델 Fine-Tuning 실시
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - generator              (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator
# - exist_dict           (dict)      : 각 모델 (CNN, StyleGAN-FineTune-v3) 의 존재 여부 (= 신규 학습 미 실시 여부)

def run_fine_tuning(generator, fine_tuning_dataloader):
    cnn_save_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v2_cnn.pth'
    stylegan_finetune_v3_exist = False

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v3 : {device}')

    # load CNN model
    cnn_model = load_cnn_model(cnn_save_path, device)

    # load or newly train Fine-Tuned Generator (StyleGAN-FineTune-v3)
    v3_gen_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v3.pth'

    try:
        fine_tuned_generator = load_stylegan_finetune_v3_model(v3_gen_path, device)
        stylegan_finetune_v3_exist = True

    except Exception as e:
        print(f'StyleGAN-FineTune-v3 model load failed : {e}')

        # train StyleGAN-FineTune-v3 model
        fine_tuned_generator = train_stylegan_finetune_v3(device, generator, fine_tuning_dataloader)
        torch.save(fine_tuned_generator.state_dict(), v3_gen_path)

    exist_dict = {'stylegan_finetune_v3': stylegan_finetune_v3_exist}
    print(f'model existance : {exist_dict}')

    return fine_tuned_generator, exist_dict