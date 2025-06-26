import torch
from torchvision.io import read_image

import os
import sys
import numpy as np
import pandas as pd

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

import stylegan.stylegan_common.stylegan_generator as gen
from stylegan.stylegan_common.visualizer import postprocess_image, save_image

from stylegan.run_stylegan_vectorfind_v7 import get_property_change_vectors as get_property_change_vectors_v7
from stylegan.run_stylegan_vectorfind_v7 import load_ohlora_z_vectors as load_ohlora_z_vectors_v7
from stylegan.run_stylegan_vectorfind_v7 import load_ohlora_w_group_names as load_ohlora_w_group_names_v7
from stylegan.run_stylegan_vectorfind_v7 import get_pm_labels as get_pm_labels_v7

from stylegan.run_stylegan_vectorfind_v8 import get_property_change_vectors as get_property_change_vectors_v8
from stylegan.run_stylegan_vectorfind_v8 import load_ohlora_z_vectors as load_ohlora_z_vectors_v8
from stylegan.run_stylegan_vectorfind_v8 import load_ohlora_w_group_names as load_ohlora_w_group_names_v8
from stylegan.run_stylegan_vectorfind_v8 import get_pm_labels as get_pm_labels_v8

from stylegan.common import (load_existing_stylegan_vectorfind_v7,
                             load_existing_stylegan_vectorfind_v8,
                             load_merged_property_score_cnn)


IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINAL_HIDDEN_DIMS_W = 512
ORIGINALLY_PROPERTY_DIMS_V7 = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값
ORIGINALLY_PROPERTY_DIMS_V8 = 7  # 원래 property (eyes, hair_color, hair_length, mouth, pose,
                                 #               background_mean, background_std) 목적으로 사용된 dimension 값
TEST_IMG_CASES = 1

GROUP_NAMES = ['hhh', 'hhl', 'hlh', 'hll', 'lhh', 'lhl', 'llh', 'lll']
PROPERTY_NAMES = ['eyes', 'mouth', 'pose']

kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)



# 옴브레 염색 적용된 이미지 생성
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_generator (nn.Module)   : StyleGAN-VectorFind-v7 or v8 의 Generator
# - eyes_vector          (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터
# - mouth_vector         (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터
# - pose_vector          (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터
# - code_w               (Tensor)      : latent code (w) 에 해당하는 부분 (dim: 512)
# - save_dir             (str)         : 이미지를 저장할 디렉토리 경로 (stylegan_vectorfind_v8/inference_test_after_training)
# - img_file_name        (str)         : 저장할 이미지 파일 이름
# - pms                  (dict)        : eyes, mouth, pose 핵심 속성 값 변화 벡터를 latent code 에 더하거나 빼기 위한 가중치
#                                         {'eyes': float, 'mouth': float, 'pose': float}
# - color                (float)       : 색상 값 (0.0 - 1.0 범위)
# - ombre_height         (float)       : 옴브레 염색 부분의 세로 길이 (0.0 - 1.0 범위)
# - ombre_grad_height    (float)       : 옴브레 염색 부분의 그라데이션 부분의 세로 길이 비율 (0.0 - 1.0 범위)

def generate_ombre_img(vectorfind_generator, eyes_vector, mouth_vector, pose_vector, code_w, save_dir, img_file_name, pms,
                       color, ombre_height, ombre_grad_height):

    def generate_image_using_w(vectorfind_generator, w, trunc_psi=1.0, trunc_layers=0, randomize_noise=False, lod=None):
        with torch.no_grad():
            wp = vectorfind_generator.truncation(w, trunc_psi, trunc_layers)
            images = vectorfind_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
            images = postprocess_image(images.detach().cpu().numpy())
        return images

    eyes_pm, mouth_pm, pose_pm = pms['eyes'], pms['mouth'], pms['pose']

    # generate image
    with torch.no_grad():
        code_w_ = code_w + eyes_pm * torch.tensor(eyes_vector[0:1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_ + mouth_pm * torch.tensor(mouth_vector[0:1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_ + pose_pm * torch.tensor(pose_vector[0:1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_.type(torch.float32)

        images = generate_image_using_w(vectorfind_generator, code_w_)

    save_image(os.path.join(save_dir, img_file_name), images[0])

    # TODO 옴브레 이미지 적용 구현


# 옴브레 염색 적용된 이미지 생성 (entrance 함수)
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_generator (nn.Module) : StyleGAN-VectorFind-v7 or v8 의 Generator
# - vectorfind_ver       (str)       : StyleGAN-VectorFind 버전 ('v7' or 'v8')
# - ohlora_no            (int)       : Oh-LoRA 이미지 번호 ('v7'의 경우 127, 672, 709, ...)
# - color                (float)     : 색상 값 (0.0 - 1.0 범위)
# - ombre_height         (float)     : 옴브레 염색 부분의 세로 길이 (0.0 - 1.0 범위)
# - ombre_grad_height    (float)     : 옴브레 염색 부분의 그라데이션 부분의 세로 길이 비율 (0.0 - 1.0 범위)

def generate_ombre_image(vectorfind_generator, vectorfind_ver, ohlora_no, color, ombre_height, ombre_grad_height):
    raise NotImplementedError


# StyleGAN-VectorFind-v7 옴브레 염색 적용 이미지 생성 테스트
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_v7_generator (nn.Module)         : StyleGAN-VectorFind-v7 의 Generator
# - eyes_vectors            (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors           (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors            (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def generate_ombre_image_using_v7(vectorfind_v7_generator, eyes_vectors, mouth_vectors, pose_vectors):
    n_vector_cnt = len(eyes_vectors['hhh'])  # equal to pre-defined SVMS_PER_EACH_PROPERTY value

    ohlora_z_vector_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_z_vectors.csv'
    ohlora_w_group_name_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_w_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors_v7(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_w_group_names = load_ohlora_w_group_names_v7(group_name_csv_path=ohlora_w_group_name_csv_path)

    # label: 'eyes', 'mouth', 'pose'
    eyes_pm_order, mouth_pm_order, pose_pm_order = get_pm_labels_v7()
    pm_cnt = len(eyes_pm_order)

    count_to_generate = len(ohlora_z_vectors)
    code_part1s_np = np.zeros((count_to_generate, ORIGINAL_HIDDEN_DIMS_Z))
    code_part2s_np = np.zeros((count_to_generate, ORIGINALLY_PROPERTY_DIMS_V7))

    # image generation
    for i in range(count_to_generate):
        print(f'generating (w/ StyleGAN-VectorFind-v7) : {i} / {count_to_generate}')

        save_dir = f'{PROJECT_DIR_PATH}/ombre/stylegan_vectorfind_v7/inference_test_after_training/test_{i:04d}'
        os.makedirs(save_dir, exist_ok=True)

        code_part1s_np[i] = ohlora_z_vectors[i][:ORIGINAL_HIDDEN_DIMS_Z]
        code_part2s_np[i] = ohlora_z_vectors[i][ORIGINAL_HIDDEN_DIMS_Z:]
        code_part1 = torch.tensor(code_part1s_np[i]).unsqueeze(0).to(torch.float32)  # 512
        code_part2 = torch.tensor(code_part2s_np[i]).unsqueeze(0).to(torch.float32)  # 3

        with torch.no_grad():
            code_w = vectorfind_v7_generator.mapping(code_part1.cuda(), code_part2.cuda())['w'].detach().cpu()

        for vi in range(n_vector_cnt):
            n_vector_idx = i * n_vector_cnt + vi
            group_name = ohlora_w_group_names[n_vector_idx]

            eyes_vector = eyes_vectors[group_name]
            mouth_vector = mouth_vectors[group_name]
            pose_vector = pose_vectors[group_name]

            for pm_idx in range(pm_cnt):
                img_file_name = f'case_{i:03d}_{vi:03d}_pm_{pm_idx:03d}.jpg'
                pms = {'eyes': eyes_pm_order[pm_idx], 'mouth': mouth_pm_order[pm_idx], 'pose': pose_pm_order[pm_idx]}

                generate_ombre_img(vectorfind_v7_generator, eyes_vector, mouth_vector, pose_vector, code_w, save_dir,
                                   img_file_name, pms, color=0.0, ombre_height=0.3, ombre_grad_height=0.4)


# StyleGAN-VectorFind-v7 옴브레 염색 적용 이미지 생성 테스트 (모델 로딩을 포함한 전 과정)
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - 없음

def generate_ombre_image_using_v7_all_process():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v1 : {device}')

    vectorfind_v7_generator = gen.StyleGANGeneratorForV6(resolution=IMAGE_RESOLUTION)  # v6, v7 Generator 는 동일한 구조

    # loading StyleGAN-VectorFind-v7 pre-trained model
    generator_state_dict = load_existing_stylegan_vectorfind_v7(device)
    vectorfind_v7_generator.load_state_dict(generator_state_dict)
    print('Existing StyleGAN-VectorFind-v7 Generator load successful!! 😊')

    # get property score changing vector
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors_v7()
    print('Existing "Property Score Changing Vector" info load successful!! 😊')

    # image generation test
    vectorfind_v7_generator.to(device)

    generate_ombre_image_using_v7(vectorfind_v7_generator,
                                  eyes_vectors,
                                  mouth_vectors,
                                  pose_vectors)


# StyleGAN-VectorFind-v8 옴브레 염색 적용 이미지 생성 테스트
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_v8_generator (nn.Module)         : StyleGAN-VectorFind-v8 의 Generator
# - eyes_vectors            (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors           (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors            (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def generate_ombre_image_using_v8(vectorfind_v8_generator, eyes_vectors, mouth_vectors, pose_vectors):
    n_vector_cnt = len(eyes_vectors['hhhh'])  # equal to pre-defined SVMS_PER_EACH_PROPERTY value

    ohlora_z_vector_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_z_vectors.csv'
    ohlora_w_group_name_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_w_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors_v8(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_w_group_names = load_ohlora_w_group_names_v8(group_name_csv_path=ohlora_w_group_name_csv_path)

    # label: 'eyes', 'mouth', 'pose'
    eyes_pm_order, mouth_pm_order, pose_pm_order = get_pm_labels_v8()
    pm_cnt = len(eyes_pm_order)

    count_to_generate = len(ohlora_z_vectors)
    code_part1s_np = np.zeros((count_to_generate, ORIGINAL_HIDDEN_DIMS_Z))
    code_part2s_np = np.zeros((count_to_generate, ORIGINALLY_PROPERTY_DIMS_V8))

    # image generation
    for i in range(count_to_generate):
        print(f'generating (w/ StyleGAN-VectorFind-v8) : {i} / {count_to_generate}')

        save_dir = f'{PROJECT_DIR_PATH}/ombre/stylegan_vectorfind_v8/inference_test_after_training/test_{i:04d}'
        os.makedirs(save_dir, exist_ok=True)

        code_part1s_np[i] = ohlora_z_vectors[i][:ORIGINAL_HIDDEN_DIMS_Z]
        code_part2s_np[i] = ohlora_z_vectors[i][ORIGINAL_HIDDEN_DIMS_Z:]
        code_part1 = torch.tensor(code_part1s_np[i]).unsqueeze(0).to(torch.float32)  # 512
        code_part2 = torch.tensor(code_part2s_np[i]).unsqueeze(0).to(torch.float32)  # 7

        with torch.no_grad():
            code_w = vectorfind_v8_generator.mapping(code_part1.cuda(), code_part2.cuda())['w'].detach().cpu()

        for vi in range(n_vector_cnt):
            n_vector_idx = i * n_vector_cnt + vi
            group_name = ohlora_w_group_names[n_vector_idx]

            eyes_vector = eyes_vectors[group_name]
            mouth_vector = mouth_vectors[group_name]
            pose_vector = pose_vectors[group_name]

            for pm_idx in range(pm_cnt):
                img_file_name = f'case_{i:03d}_{vi:03d}_pm_{pm_idx:03d}.jpg'
                pms = {'eyes': eyes_pm_order[pm_idx], 'mouth': mouth_pm_order[pm_idx], 'pose': pose_pm_order[pm_idx]}

                generate_ombre_img(vectorfind_v8_generator, eyes_vector, mouth_vector, pose_vector, code_w, save_dir,
                                   img_file_name, pms, color=0.0, ombre_height=0.3, ombre_grad_height=0.4)


# StyleGAN-VectorFind-v8 옴브레 염색 적용 이미지 생성 테스트 (모델 로딩을 포함한 전 과정)
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - 없음

def generate_ombre_image_using_v8_all_process():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v8 : {device}')

    finetune_v8_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)  # v1, v8 Generator 는 동일한 구조

    # loading StyleGAN-VectorFind-v8 pre-trained model
    generator_state_dict = load_existing_stylegan_vectorfind_v8(device)
    finetune_v8_generator.load_state_dict(generator_state_dict)
    print('Existing StyleGAN-VectorFind-v8 Generator load successful!! 😊')

    # get property score changing vector
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors_v8(vectorfind_version='v8')
    print('Existing "Property Score Changing Vector" info load successful!! 😊')

    # image generation test
    finetune_v8_generator.to(device)

    generate_ombre_image_using_v8(finetune_v8_generator,
                                  eyes_vectors,
                                  mouth_vectors,
                                  pose_vectors)
