import torch
import cv2

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from stylegan.run_stylegan_vectorfind_v6 import (load_ohlora_z_vectors,
                                                 load_ohlora_z_group_names,
                                                 get_property_change_vectors)
from stylegan.common import load_existing_stylegan_vectorfind_v6
import stylegan.stylegan_common.stylegan_generator as gen

from generate_ohlora_image import generate_images


IMAGE_RESOLUTION = 256


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = f'{PROJECT_DIR_PATH}/final_product/ohlora_images'
    os.makedirs(img_path, exist_ok=True)

    # load property score change vectors
    ohlora_z_vector_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/ohlora_z_vectors.csv'
    ohlora_z_group_name_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/ohlora_z_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_z_group_names = load_ohlora_z_group_names(group_name_csv_path=ohlora_z_group_name_csv_path)

    # load StyleGAN-VectorFind-v6 generator
    vectorfind_v6_generator = gen.StyleGANGeneratorForV6(resolution=IMAGE_RESOLUTION)
    generator_state_dict = load_existing_stylegan_vectorfind_v6(device)
    vectorfind_v6_generator.load_state_dict(generator_state_dict)
    vectorfind_v6_generator.to(device)

    # generate images
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors()

    for idx, (ohlora_z_vector, ohlora_z_group_name) in enumerate(zip(ohlora_z_vectors, ohlora_z_group_names)):
        eyes_vector = eyes_vectors[ohlora_z_group_name][0]
        mouth_vector = mouth_vectors[ohlora_z_group_name][0]
        pose_vector = pose_vectors[ohlora_z_group_name][0]

        ohlora_image = generate_images(vectorfind_v6_generator, ohlora_z_vector,
                                       eyes_vector, mouth_vector, pose_vector,
                                       eyes_pm=0.0, mouth_pm=0.0, pose_pm=0.0)

        img_save_path = f'{img_path}/{idx}.png'
        cv2.imwrite(img_save_path, ohlora_image[:, :, ::-1])
