
import cv2
import numpy as np
import torch

import os
import sys

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from ombre.generate_ombre_images import generate_ombre_img


ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS_V7 = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값
ORIGINALLY_PROPERTY_DIMS_V8 = 7  # 원래 property (eyes, hair_color, hair_length, mouth, pose,
                                 #               background_mean, background_std) 목적으로 사용된 dimension 값


# Oh-LoRA (오로라) 이미지 생성 및 표시
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - vectorfind_generator (nn.Module)   : StyleGAN-VectorFind-v7 또는 StyleGAN-VectorFind-v8 의 Generator
# - ohlora_z_vector      (NumPy array) : Oh-LoRA 이미지 생성용 latent z vector, dim = (512 + 3,) (v7) or (512 + 7,) (v8)
# - eyes_vector          (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - mouth_vector         (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - pose_vector          (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터, dim = (512,)
# - eyes_score           (float)       : 눈을 뜬 정도 (eyes) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)
# - mouth_score          (float)       : 입을 벌린 정도 (mouth) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)
# - pose_score           (float)       : 고개 돌림 (pose) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)

# Returns :
# - 직접 반환되는 값 없음
# - final_product/ohlora.png 경로에 오로라 이미지 생성 및 화면에 display

def generate_and_show_ohlora_image(vectorfind_generator, hair_seg_model, vectorfind_ver,
                                   ohlora_z_vector, eyes_vector, mouth_vector, pose_vector,
                                   eyes_score, mouth_score, pose_score,
                                   color, ombre_height, ombre_grad_height):

    code_part1s_np = np.zeros((1, ORIGINAL_HIDDEN_DIMS_Z))
    if vectorfind_ver == 'v7':
        code_part2s_np = np.zeros((1, ORIGINALLY_PROPERTY_DIMS_V7))
    else:  # v8
        code_part2s_np = np.zeros((1, ORIGINALLY_PROPERTY_DIMS_V8))

    # image generation
    code_part1s_np[0] = ohlora_z_vector[:ORIGINAL_HIDDEN_DIMS_Z]
    code_part2s_np[0] = ohlora_z_vector[ORIGINAL_HIDDEN_DIMS_Z:]
    code_part1 = torch.tensor(code_part1s_np[0]).unsqueeze(0).to(torch.float32)  # 512
    code_part2 = torch.tensor(code_part2s_np[0]).unsqueeze(0).to(torch.float32)  # 3 (VectorFind-v7), 7 (VectorFind-v8)

    pms = {'eyes': eyes_score, 'mouth': mouth_score, 'pose': pose_score}

    with torch.no_grad():
        code_w = vectorfind_generator.mapping(code_part1.cuda(), code_part2.cuda())['w'].detach().cpu()

        ombre_image = generate_ombre_img(vectorfind_generator, hair_seg_model, eyes_vector, mouth_vector, pose_vector,
                                         code_w, save_dir=None, img_file_name=None, pms=pms,
                                         color=color, ombre_height=ombre_height, ombre_grad_height=ombre_grad_height,
                                         instruct='return')

    cv2.imshow('Oh-LoRA', ombre_image[:, :, ::-1])
    _ = cv2.waitKey(1)
