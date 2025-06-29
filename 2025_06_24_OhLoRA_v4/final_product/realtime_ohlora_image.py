import torch
import cv2
import numpy as np

import os
import sys
import time

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from stylegan.run_stylegan_vectorfind_v8 import (load_ohlora_z_vectors,
                                                 load_ohlora_w_group_names,
                                                 get_property_change_vectors)
from stylegan.common import load_existing_stylegan_vectorfind_v8
import stylegan.stylegan_common.stylegan_generator as gen

from generate_ohlora_image import generate_images


IMAGE_RESOLUTION = 256


# Oh-LoRA 👱‍♀️ (오로라) 이미지 실시간 표시 (display) 테스트
# Create Date : 2025.06.29
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = f'{PROJECT_DIR_PATH}/final_product/ohlora_images'
    os.makedirs(img_path, exist_ok=True)

    # load property score change vectors
    ohlora_z_vector_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_z_vectors.csv'
    ohlora_w_group_name_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_w_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_w_group_names = load_ohlora_w_group_names(group_name_csv_path=ohlora_w_group_name_csv_path)

    # load StyleGAN-VectorFind-v8 generator
    vectorfind_v8_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)
    generator_state_dict = load_existing_stylegan_vectorfind_v8(device)
    vectorfind_v8_generator.load_state_dict(generator_state_dict)
    vectorfind_v8_generator.to(device)

    # generate images
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors()

    for idx, (ohlora_z_vector, ohlora_w_group_name) in enumerate(zip(ohlora_z_vectors, ohlora_w_group_names)):
        eyes_vector = eyes_vectors[ohlora_w_group_name][0]
        mouth_vector = mouth_vectors[ohlora_w_group_name][0]
        pose_vector = pose_vectors[ohlora_w_group_name][0]

        # image generation test
        ohlora_image = generate_images(vectorfind_v8_generator, ohlora_z_vector,
                                       eyes_vector, mouth_vector, pose_vector,
                                       eyes_pm=0.0, mouth_pm=0.0, pose_pm=0.0)

        img_save_path = f'{img_path}/{idx}.png'
        cv2.imwrite(img_save_path, ohlora_image[:, :, ::-1])

        # image display test
        print(f'\n=== index {idx} image display test ===')
        display_realtime_ohlora_image(vectorfind_v8_generator, ohlora_z_vector, eyes_vector, mouth_vector, pose_vector)
