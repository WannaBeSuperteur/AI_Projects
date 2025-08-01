
import cv2
import numpy as np

import os
import sys
import time

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from generate_ohlora_image import generate_images


IMAGE_RESOLUTION = 256


# Oh-LoRA 👱‍♀️ (오로라) 이미지 실시간 표시 (display) 테스트
# Create Date : 2025.08.01
# Last Update Date : -

# Arguments:
# - vectorfind_generator (nn.Module)   : StyleGAN-VectorFind-v7 또는 StyleGAN-VectorFind-v8 의 Generator
# - ohlora_z_vector      (NumPy array) : Oh-LoRA 이미지 생성용 latent z vector, dim = (512 + 3,) (v7) or (512 + 7,) (v8)
# - eyes_vector          (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - mouth_vector         (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - pose_vector          (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터, dim = (512,)

def display_realtime_ohlora_image(vectorfind_generator, ohlora_z_vector, eyes_vector, mouth_vector, pose_vector):
    eyes_pms = np.linspace(-1.2, 1.2, 20)
    mouth_pms = np.linspace(-1.8, 1.8, 20)
    pose_pms = np.linspace(-1.8, 0.6, 20)

    pms_list = [eyes_pms, mouth_pms, pose_pms]
    property_names = ['eyes', 'mouth', 'pose']

    for pms, property_name in zip(pms_list, property_names):
        for pm in pms:
            print(f'time: {time.time()}, property: {property_name}, pm: {pm}')

            eyes_pm = pm if property_name == 'eyes' else 0.0
            mouth_pm = pm if property_name == 'mouth' else 0.0
            pose_pm = pm if property_name == 'pose' else 0.0

            ohlora_image_to_display = generate_images(vectorfind_generator, ohlora_z_vector,
                                                      eyes_vector, mouth_vector, pose_vector,
                                                      eyes_pm=eyes_pm, mouth_pm=mouth_pm, pose_pm=pose_pm)

            cv2.imshow('Oh-LoRA', ohlora_image_to_display[:, :, ::-1])
            _ = cv2.waitKey(10)

