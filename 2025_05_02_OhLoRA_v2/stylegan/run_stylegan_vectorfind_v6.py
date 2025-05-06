from stylegan_vectorfind_v6.main import main as stylegan_vectorfind_v6_main
from common import load_existing_stylegan_finetune_v1
import stylegan_common.stylegan_generator as gen

import torch
import os
import numpy as np
import pandas as pd

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256


# Property Score 값을 변경하기 위해 latent vector z 에 가감할 벡터 정보 반환
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - eyes_vector (NumPy Array) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보
# - mouth_vector (NumPy Array) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보
# - pose_vector (NumPy Array) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보

def get_property_change_vectors():
    vector_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/property_score_vectors'

    eyes_vector = np.array(pd.read_csv(f'{vector_save_dir}/eyes_change_z_vector.csv', index_col=0))
    mouth_vector = np.array(pd.read_csv(f'{vector_save_dir}/mouth_change_z_vector.csv', index_col=0))
    pose_vector = np.array(pd.read_csv(f'{vector_save_dir}/pose_change_z_vector.csv', index_col=0))

    return eyes_vector, mouth_vector, pose_vector


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v1 : {device}')

    finetune_v1_generator = gen.StyleGANGeneratorForV6(resolution=IMAGE_RESOLUTION)
    generator_state_dict = load_existing_stylegan_finetune_v1(device)
    print('Existing StyleGAN-FineTune-v1 Generator load successful!! 😊')

    # load state dict (generator)
    del generator_state_dict['mapping.label_weight']  # size mismatch because of modified property vector dim (7 -> 3)
    finetune_v1_generator.load_state_dict(generator_state_dict, strict=False)

    # get property score changing vector
    try:
        eyes_vector, mouth_vector, pose_vector = get_property_change_vectors()

    except:
        stylegan_vectorfind_v6_main(finetune_v1_generator, device)
        eyes_vector, mouth_vector, pose_vector = get_property_change_vectors()

    # image generation test
    # TODO: implement

