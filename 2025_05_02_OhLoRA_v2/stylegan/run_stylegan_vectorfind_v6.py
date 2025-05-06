from stylegan_vectorfind_v6.main import main as stylegan_vectorfind_v6_main
from common import load_existing_stylegan_finetune_v1
import stylegan_common.stylegan_generator as gen
from stylegan_common.visualizer import postprocess_image, save_image

import torch
import os
import numpy as np
import pandas as pd

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256

ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS_Z = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값
TEST_IMG_CASES = 20


# Property Score 값을 변경하기 위해 latent vector z 에 가감할 벡터 정보 반환
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - eyes_vector  (NumPy Array) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보
# - mouth_vector (NumPy Array) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보
# - pose_vector  (NumPy Array) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보

def get_property_change_vectors():
    vector_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/property_score_vectors'

    eyes_vector = np.array(pd.read_csv(f'{vector_save_dir}/eyes_change_z_vector.csv', index_col=0))
    mouth_vector = np.array(pd.read_csv(f'{vector_save_dir}/mouth_change_z_vector.csv', index_col=0))
    pose_vector = np.array(pd.read_csv(f'{vector_save_dir}/pose_change_z_vector.csv', index_col=0))

    return eyes_vector, mouth_vector, pose_vector


# latent vector z 에 가감할 Property Score Vector 를 이용한 Property Score 값 변화 테스트 (이미지 생성 테스트)
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module)   : StyleGAN-FineTune-v1 의 Generator
# - eyes_vector           (NumPy Array) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보
# - mouth_vector          (NumPy Array) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보
# - pose_vector           (NumPy Array) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보

# Returns:
# - stylegan_vectorfind_v6/inference_test_after_training 디렉토리에 이미지 생성 결과 저장

def run_image_generation_test(finetune_v1_generator, eyes_vector, mouth_vector, pose_vector):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/inference_test_after_training'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(TEST_IMG_CASES):
        code_part1 = torch.randn(1, ORIGINAL_HIDDEN_DIMS_Z)      # 512
        code_part2 = torch.randn(1, ORIGINALLY_PROPERTY_DIMS_Z)  # 3

        vector_names = ['eyes', 'mouth', 'pose']
        vectors = [eyes_vector, mouth_vector, pose_vector]

        images = finetune_v1_generator(code_part1.cuda(), code_part2.cuda(), **kwargs_val)['image']
        images = postprocess_image(images.detach().cpu().numpy())
        save_image(os.path.join(save_dir, f'case_{i:02d}_as_original.jpg'), images[0])

        for vector_name, vector in zip(vector_names, vectors):
            pms = [-3.0, -1.0, 1.0, 3.0]

            for pm_idx, pm in enumerate(pms):
                with torch.no_grad():
                    code_part1_ = code_part1 + pm * torch.tensor(vector[:, :ORIGINAL_HIDDEN_DIMS_Z])  # 512
                    code_part2_ = code_part2 + pm * torch.tensor(vector[:, ORIGINAL_HIDDEN_DIMS_Z:])  # 3
                    code_part1_ = code_part1_.type(torch.float32)
                    code_part2_ = code_part2_.type(torch.float32)

                    images = finetune_v1_generator(code_part1_.cuda(), code_part2_.cuda(), **kwargs_val)['image']
                    images = postprocess_image(images.detach().cpu().numpy())
                    save_image(os.path.join(save_dir, f'case_{i:02d}_{vector_name}_pm_{pm_idx}.jpg'), images[0])


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
    finetune_v1_generator.to(device)
    run_image_generation_test(finetune_v1_generator, eyes_vector, mouth_vector, pose_vector)
