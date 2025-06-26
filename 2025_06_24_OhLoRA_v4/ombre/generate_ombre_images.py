import torch
import os
import sys
import numpy as np
import pandas as pd

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

import stylegan.stylegan_common.stylegan_generator as gen
from stylegan.stylegan_common.visualizer import postprocess_image, save_image
from stylegan.property_score_cnn import load_cnn_model as load_property_cnn_model
from stylegan.run_stylegan_vectorfind_v7 import get_property_change_vectors as get_property_change_vectors_v7
from stylegan.run_stylegan_vectorfind_v8 import get_property_change_vectors as get_property_change_vectors_v8

from stylegan.common import (stylegan_transform,
                             load_existing_stylegan_finetune_v1,
                             load_existing_stylegan_vectorfind_v7,
                             load_existing_stylegan_finetune_v8,
                             load_existing_stylegan_vectorfind_v8,
                             load_merged_property_score_cnn)


IMAGE_RESOLUTION = 256

ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINAL_HIDDEN_DIMS_W = 512
ORIGINALLY_PROPERTY_DIMS = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값

TEST_IMG_CASES = 1

OHLORA_FINAL_VECTORS_TEST_REPORT_PATH = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/final_vector_test_report'
os.makedirs(OHLORA_FINAL_VECTORS_TEST_REPORT_PATH, exist_ok=True)

GROUP_NAMES = ['hhh', 'hhl', 'hlh', 'hll', 'lhh', 'lhl', 'llh', 'lll']
PROPERTY_NAMES = ['eyes', 'mouth', 'pose']

kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)



# 옴브레 염색 적용된 이미지 생성
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_ver    (str)   : StyleGAN-VectorFind 버전 ('v7' or 'v8')
# - ohlora_no         (int)   : Oh-LoRA 이미지 번호 ('v7'의 경우 127, 672, 709, ...)
# - color             (float) : 색상 값 (0.0 - 1.0 범위)
# - ombre_height      (float) : 옴브레 염색 부분의 세로 길이 (0.0 - 1.0 범위)
# - ombre_grad_height (float) : 옴브레 염색 부분의 그라데이션 부분의 세로 길이 비율 (0.0 - 1.0 범위)

def generate_ombre_image(vectorfind_ver, ohlora_no, color, ombre_height):
    raise NotImplementedError


# StyleGAN-VectorFind-v7 옴브레 염색 적용 이미지 생성 테스트
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_v7_generator (nn.Module)         : StyleGAN-VectorFind-v7 의 Generator
# - property_score_cnn      (nn.Module)         : 핵심 속성 값을 계산하기 위한 CNN
# - eyes_vectors            (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors           (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors            (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def generate_ombre_image_using_v7(vectorfind_v7_generator, property_score_cnn, eyes_vectors, mouth_vectors, pose_vectors):
    raise NotImplementedError


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

    # get Property Score CNN
    property_cnn_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_score_cnn = load_property_cnn_model(property_cnn_path, device)
    print('Existing Property Score CNN load successful!! 😊')

    # image generation test
    vectorfind_v7_generator.to(device)

    generate_ombre_image_using_v7(vectorfind_v7_generator,
                                  property_score_cnn,
                                  eyes_vectors,
                                  mouth_vectors,
                                  pose_vectors)


# StyleGAN-VectorFind-v8 옴브레 염색 적용 이미지 생성 테스트
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_v8_generator (nn.Module)         : StyleGAN-VectorFind-v8 의 Generator
# - property_score_cnn      (nn.Module)         : 핵심 속성 값을 계산하기 위한 CNN
# - eyes_vectors            (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors           (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors            (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def generate_ombre_image_using_v8(vectorfind_v8_generator, property_score_cnn, eyes_vectors, mouth_vectors, pose_vectors):
    raise NotImplementedError


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

    # get Merged Property Score CNN
    property_score_cnn = load_merged_property_score_cnn(device)

    # image generation test
    finetune_v8_generator.to(device)

    generate_ombre_image_using_v8(finetune_v8_generator,
                                  property_score_cnn,
                                  eyes_vectors,
                                  mouth_vectors,
                                  pose_vectors)
