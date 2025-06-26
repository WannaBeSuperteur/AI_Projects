from generate_ombre_images import generate_ombre_image
from common import get_property_score_change_vectors

from stylegan.run_stylegan_vectorfind_v7 import load_ohlora_z_vectors as load_ohlora_z_vectors_v7
from stylegan.run_stylegan_vectorfind_v7 import load_ohlora_w_group_names as load_ohlora_w_group_names_v7
from stylegan.run_stylegan_vectorfind_v8 import load_ohlora_z_vectors as load_ohlora_z_vectors_v8
from stylegan.run_stylegan_vectorfind_v8 import load_ohlora_w_group_names as load_ohlora_w_group_names_v8

import imageio
import numpy as np

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

OHLORA_Z_VECTOR_CSV_PATH_V7 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_z_vectors.csv'
OHLORA_W_GROUP_NAME_CSV_PATH_V7 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_w_group_names.csv'
OHLORA_Z_VECTOR_CSV_PATH_V8 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_z_vectors.csv'
OHLORA_W_GROUP_NAME_CSV_PATH_V8 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_w_group_names.csv'

CASE_NO_TO_IDX_V7 = { 127:  0,  672:  1,  709:  2,  931:  3, 1017:  4, 1073:  5, 1162:  6, 1211:  7, 1277:  8, 1351:  9,
                     1359: 10, 1409: 11, 1591: 12, 1646: 13, 1782: 14, 1788: 15, 1819: 16, 1836: 17, 1905: 18, 1918: 19,
                     2054: 20, 2089: 21, 2100: 22, 2111: 23, 2137: 24, 2185: 25, 2240: 26}

CASE_NO_TO_IDX_V8 = {  83:  0,  143:  1,  194:  2,  214:  3,  285:  4,  483:  5,  536:  6,  679:  7,  853:  8,  895:  9,
                      986: 10,  991: 11, 1064: 12, 1180: 13, 1313: 14, 1535: 15, 1750: 16, 1792: 17, 1996: 18}


# GIF 이미지 생성
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_generator   (nn.Module)         : StyleGAN-VectorFind-v7 or v8 의 Generator
# - hair_seg_model         (nn.Module)         : Hair 영역 추출용 Segmentation Model
# - vectorfind_ver         (str)               : StyleGAN-VectorFind 버전 ('v7' or 'v8')
# - ohlora_no              (int)               : Oh-LoRA 이미지 번호 ('v7'의 경우 127, 672, 709, ...)
# - color_list             (list(float))       : 색상 값 (0.0 - 1.0 범위) 의 list
# - ombre_height_list      (list(float))       : 옴브레 염색 부분의 세로 길이 (0.0 - 1.0 범위) 의 list
# - ombre_grad_height_list (list(float))       : 옴브레 염색 부분의 그라데이션 부분의 세로 길이 비율 (0.0 - 1.0 범위) 의 list
# - pms_list               (dict(list(float))) : eyes, mouth, pose 핵심 속성 값 변화 벡터를 latent code 에 더하거나 빼기 위한
#                                                가중치의 list
#                                                {'eyes': list(float), 'mouth': list(float), 'pose': list(float)}
# - gif_save_path          (str)               : GIF 이미지의 저장 경로
# - duration               (float)             : 프레임 당 시간 (초)

def generate_gif(vectorfind_generator, hair_seg_model, vectorfind_ver, ohlora_no,
                 color_list, ombre_height_list, ombre_grad_height_list, pms_list, gif_save_path, duration=0.05):

    print(f'generating GIF image for {vectorfind_ver} / {ohlora_no} ...')
    eyes_vectors, mouth_vectors, pose_vectors = get_property_score_change_vectors(vectorfind_ver)
    ohlora_images = []
    n_frames = len(color_list)

    if vectorfind_ver == 'v7':
        ohlora_z_vectors = load_ohlora_z_vectors_v7(vector_csv_path=OHLORA_Z_VECTOR_CSV_PATH_V7)
        ohlora_w_group_names = load_ohlora_w_group_names_v7(group_name_csv_path=OHLORA_W_GROUP_NAME_CSV_PATH_V7)
        ohlora_idx = CASE_NO_TO_IDX_V7[ohlora_no]
    else:  # v8
        ohlora_z_vectors = load_ohlora_z_vectors_v8(vector_csv_path=OHLORA_Z_VECTOR_CSV_PATH_V8)
        ohlora_w_group_names = load_ohlora_w_group_names_v8(group_name_csv_path=OHLORA_W_GROUP_NAME_CSV_PATH_V8)
        ohlora_idx = CASE_NO_TO_IDX_V8[ohlora_no]

    for idx in range(n_frames):
        color = color_list[idx]
        ombre_height = ombre_height_list[idx]
        ombre_grad_height = ombre_grad_height_list[idx]
        pms = {'eyes': pms_list['eyes'][idx], 'mouth': pms_list['mouth'][idx], 'pose': pms_list['pose'][idx]}

        ohlora_image = generate_ombre_image(vectorfind_generator, hair_seg_model,
                                            eyes_vectors, mouth_vectors, pose_vectors,
                                            vectorfind_ver, color, ombre_height, ombre_grad_height, pms,
                                            ohlora_z_vectors, ohlora_w_group_names, ohlora_idx)
        ohlora_images.append(ohlora_image.astype(np.uint8))

    imageio.mimsave(gif_save_path, ohlora_images, duration=duration, loop=0)


