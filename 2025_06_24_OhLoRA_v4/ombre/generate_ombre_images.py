import torch
import torchvision.transforms as transforms

import os
import sys

import numpy as np
import cv2

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

from hue_added_colors import get_hue_added_colors_list


IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINAL_HIDDEN_DIMS_W = 512
ORIGINALLY_PROPERTY_DIMS_V7 = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값
ORIGINALLY_PROPERTY_DIMS_V8 = 7  # 원래 property (eyes, hair_color, hair_length, mouth, pose,
                                 #               background_mean, background_std) 목적으로 사용된 dimension 값
TEST_IMG_CASES = 1

OHLORA_Z_VECTOR_CSV_PATH_V7 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_z_vectors.csv'
OHLORA_W_GROUP_NAME_CSV_PATH_V7 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_w_group_names.csv'
OHLORA_Z_VECTOR_CSV_PATH_V8 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_z_vectors.csv'
OHLORA_W_GROUP_NAME_CSV_PATH_V8 = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_w_group_names.csv'

CASE_NO_TO_IDX_V7 = { 127:  0,  672:  1,  709:  2,  931:  3, 1017:  4, 1073:  5, 1162:  6, 1211:  7, 1277:  8, 1351:  9,
                     1359: 10, 1409: 11, 1591: 12, 1646: 13, 1782: 14, 1788: 15, 1819: 16, 1836: 17, 1905: 18, 1918: 19,
                     2054: 20, 2089: 21, 2100: 22, 2111: 23, 2137: 24, 2185: 25, 2240: 26}

CASE_NO_TO_IDX_V8 = {  83:  0,  143:  1,  194:  2,  214:  3,  285:  4,  483:  5,  536:  6,  679:  7,  853:  8,  895:  9,
                      986: 10,  991: 11, 1064: 12, 1180: 13, 1313: 14, 1535: 15, 1750: 16, 1792: 17, 1996: 18}

GROUP_NAMES = ['hhh', 'hhl', 'hlh', 'hll', 'lhh', 'lhl', 'llh', 'lll']
PROPERTY_NAMES = ['eyes', 'mouth', 'pose']

kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)

seg_model_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

hue_added_colors = get_hue_added_colors_list()


def generate_image_using_w(vectorfind_generator, w, trunc_psi=1.0, trunc_layers=0, randomize_noise=False, lod=None):
    with torch.no_grad():
        wp = vectorfind_generator.truncation(w, trunc_psi, trunc_layers)
        images = vectorfind_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
        images = postprocess_image(images.detach().cpu().numpy())
    return images


# 옴브레 염색 적용된 이미지 생성
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_generator (nn.Module)   : StyleGAN-VectorFind-v7 or v8 의 Generator
# - hair_seg_model       (nn.Module)   : Hair 영역 추출용 Segmentation Model
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
# - instruct             (str)         : 'save' (이미지 저장) or 'return' (이미지 반환)

# Returns:
# - ombre_image (Numpy array) : 옴브레 염색 적용된 이미지 (instruct == 'return' 일 때)

def generate_ombre_img(vectorfind_generator, hair_seg_model, eyes_vector, mouth_vector, pose_vector, code_w, save_dir,
                       img_file_name, pms, color=0.0, ombre_height=0.3, ombre_grad_height=0.4, instruct='save'):

    eyes_pm, mouth_pm, pose_pm = pms['eyes'], pms['mouth'], pms['pose']

    # generate image
    with torch.no_grad():
        code_w_ = code_w + eyes_pm * torch.tensor(eyes_vector[0:1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_ + mouth_pm * torch.tensor(mouth_vector[0:1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_ + pose_pm * torch.tensor(pose_vector[0:1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_.type(torch.float32)

        images = generate_image_using_w(vectorfind_generator, code_w_)

    ombre_image = apply_ombre(hair_seg_model, images[0], color, ombre_height, ombre_grad_height)

    if instruct == 'save':
        save_image(os.path.join(save_dir, img_file_name), ombre_image)

    elif instruct == 'return':
        return ombre_image


# 옴브레 스타일 적용
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - hair_seg_model    (nn.Module)   : Hair 영역 추출용 Segmentation Model
# - original_image    (Numpy array) : 옴브레 스타일 적용 이전 이미지
# - color             (float)       : 색상 값 (0.0 - 1.0 범위)
# - ombre_height      (float)       : 옴브레 염색 부분의 세로 길이 (0.0 - 1.0 범위)
# - ombre_grad_height (float)       : 옴브레 염색 부분의 그라데이션 부분의 세로 길이 비율 (0.0 - 1.0 범위)

# Returns:
# - ombre_image (Numpy array) : 옴브레 스타일 적용된 이미지

def apply_ombre(hair_seg_model, original_image, color, ombre_height, ombre_grad_height):
    global hue_added_colors

    original_image_tensor = seg_model_transform(original_image)

    # load image & generate hair area map
    hair_area_map = hair_seg_model(original_image_tensor.unsqueeze(0).cuda())
    hair_area_map = hair_area_map.detach().cpu().numpy()
    hair_area_map = hair_area_map[0, 0, :, :] * 255.0
    hair_area_map = np.expand_dims(hair_area_map, axis=2)
    hair_area_map = cv2.resize(hair_area_map, dsize=(IMAGE_RESOLUTION, IMAGE_RESOLUTION), interpolation=cv2.INTER_CUBIC)

    hair_area_map = (hair_area_map - 0.75 * 255.0) * 4.0
    hair_area_map = np.clip(hair_area_map, 0.0, 255.0)

    # compute top & bottom y-axis of hair area
    hair_top_y = None
    hair_bottom_y = None

    for y in range(IMAGE_RESOLUTION):
        if max(hair_area_map[y]) >= 0.75 * 255:
            if hair_top_y is None:
                hair_top_y = y
            hair_bottom_y = y

    # decide area to apply ombre style
    if hair_top_y is not None and hair_bottom_y is not None:
        ombre_grad_top_y = int(hair_top_y + (hair_bottom_y - hair_top_y) * (1.0 - ombre_height))
        ombre_grad_bottom_y = int(ombre_grad_top_y + (hair_bottom_y - ombre_grad_top_y) * ombre_grad_height)

        weights = np.zeros(IMAGE_RESOLUTION)
        weights[:ombre_grad_top_y] = 0.0
        weights[ombre_grad_top_y:ombre_grad_bottom_y] = (
                np.arange(ombre_grad_bottom_y - ombre_grad_top_y) / (ombre_grad_bottom_y - ombre_grad_top_y))
        weights[ombre_grad_bottom_y:] = 1.0

        hair_area_map = hair_area_map * weights[:, np.newaxis]

    # apply ombre style
    hue_value_360 = int(color * 360.0)
    hue_added_color = 0.35 * original_image + 0.25 * np.array(hue_added_colors[hue_value_360]) - 0.15 * 255.0
    color_added_image = np.clip(original_image + hue_added_color, 0, 255).astype(np.uint8)

    # generate final ombre image
    hair_area_map_ = np.expand_dims(hair_area_map, axis=2) / 255.0
    ombre_image = (1.0 - hair_area_map_) * original_image + hair_area_map_ * color_added_image

    return ombre_image


# 옴브레 염색 적용된 이미지 생성 (entrance 함수)
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_generator (nn.Module)         : StyleGAN-VectorFind-v7 or v8 의 Generator
# - hair_seg_model       (nn.Module)         : Hair 영역 추출용 Segmentation Model
# - eyes_vectors         (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors        (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors         (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - vectorfind_ver       (str)               : StyleGAN-VectorFind 버전 ('v7' or 'v8')
# - ohlora_no            (int)               : Oh-LoRA 이미지 번호 ('v7'의 경우 127, 672, 709, ...)
# - color                (float)             : 색상 값 (0.0 - 1.0 범위)
# - ombre_height         (float)             : 옴브레 염색 부분의 세로 길이 (0.0 - 1.0 범위)
# - ombre_grad_height    (float)             : 옴브레 염색 부분의 그라데이션 부분의 세로 길이 비율 (0.0 - 1.0 범위)
# - pms                  (dict)              : 핵심 속성 값 가감 가중치
#                                              {'eyes': float, 'mouth': float, 'pose': float}

def generate_ombre_image(vectorfind_generator, hair_seg_model, eyes_vectors, mouth_vectors, pose_vectors,
                         vectorfind_ver, ohlora_no, color, ombre_height, ombre_grad_height, pms):

    if vectorfind_ver == 'v7':
        ohlora_z_vectors = load_ohlora_z_vectors_v7(vector_csv_path=OHLORA_Z_VECTOR_CSV_PATH_V7)
        ohlora_w_group_names = load_ohlora_w_group_names_v7(group_name_csv_path=OHLORA_W_GROUP_NAME_CSV_PATH_V7)
        ohlora_idx = CASE_NO_TO_IDX_V7[ohlora_no]
    else:  # v8
        ohlora_z_vectors = load_ohlora_z_vectors_v8(vector_csv_path=OHLORA_Z_VECTOR_CSV_PATH_V8)
        ohlora_w_group_names = load_ohlora_w_group_names_v8(group_name_csv_path=OHLORA_W_GROUP_NAME_CSV_PATH_V8)
        ohlora_idx = CASE_NO_TO_IDX_V8[ohlora_no]

    code_part1s_np = np.zeros((1, ORIGINAL_HIDDEN_DIMS_Z))
    if vectorfind_ver == 'v7':
        code_part2s_np = np.zeros((1, ORIGINALLY_PROPERTY_DIMS_V7))
    else:  # v8
        code_part2s_np = np.zeros((1, ORIGINALLY_PROPERTY_DIMS_V8))

    # image generation
    code_part1s_np[0] = ohlora_z_vectors[ohlora_idx][:ORIGINAL_HIDDEN_DIMS_Z]
    code_part2s_np[0] = ohlora_z_vectors[ohlora_idx][ORIGINAL_HIDDEN_DIMS_Z:]
    code_part1 = torch.tensor(code_part1s_np[0]).unsqueeze(0).to(torch.float32)  # 512
    code_part2 = torch.tensor(code_part2s_np[0]).unsqueeze(0).to(torch.float32)  # 3 (VectorFind-v7), 7 (VectorFind-v8)

    with torch.no_grad():
        code_w = vectorfind_generator.mapping(code_part1.cuda(), code_part2.cuda())['w'].detach().cpu()

        group_name = ohlora_w_group_names[ohlora_idx]
        eyes_vector = eyes_vectors[group_name]
        mouth_vector = mouth_vectors[group_name]
        pose_vector = pose_vectors[group_name]

        ombre_image = generate_ombre_img(vectorfind_generator, hair_seg_model, eyes_vector, mouth_vector, pose_vector,
                                         code_w, save_dir=None, img_file_name=None, pms=pms,
                                         color=color, ombre_height=ombre_height, ombre_grad_height=ombre_grad_height,
                                         instruct='return')

    return ombre_image


# StyleGAN-VectorFind-v7 옴브레 염색 적용 이미지 생성 테스트
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_v7_generator (nn.Module)         : StyleGAN-VectorFind-v7 의 Generator
# - hair_seg_model          (nn.Module)         : Hair 영역 추출용 Segmentation Model
# - eyes_vectors            (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors           (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors            (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def generate_ombre_image_using_v7(vectorfind_v7_generator, hair_seg_model, eyes_vectors, mouth_vectors, pose_vectors):
    n_vector_cnt = len(eyes_vectors['hhh'])  # equal to pre-defined SVMS_PER_EACH_PROPERTY value

    ohlora_z_vectors = load_ohlora_z_vectors_v7(vector_csv_path=OHLORA_Z_VECTOR_CSV_PATH_V7)
    ohlora_w_group_names = load_ohlora_w_group_names_v7(group_name_csv_path=OHLORA_W_GROUP_NAME_CSV_PATH_V7)

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

                generate_ombre_img(vectorfind_v7_generator, hair_seg_model, eyes_vector, mouth_vector, pose_vector,
                                   code_w, save_dir, img_file_name, pms,
                                   color=(pm_idx / pm_cnt), ombre_height=0.4, ombre_grad_height=0.6)


# StyleGAN-VectorFind-v7 옴브레 염색 적용 이미지 생성 테스트 (모델 로딩을 포함한 전 과정)
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - hair_seg_model (nn.Module) : Hair 영역 추출용 Segmentation Model

def generate_ombre_image_using_v7_all_process(hair_seg_model):
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
                                  hair_seg_model,
                                  eyes_vectors,
                                  mouth_vectors,
                                  pose_vectors)


# StyleGAN-VectorFind-v8 옴브레 염색 적용 이미지 생성 테스트
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_v8_generator (nn.Module)         : StyleGAN-VectorFind-v8 의 Generator
# - hair_seg_model          (nn.Module)         : Hair 영역 추출용 Segmentation Model
# - eyes_vectors            (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors           (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors            (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def generate_ombre_image_using_v8(vectorfind_v8_generator, hair_seg_model, eyes_vectors, mouth_vectors, pose_vectors):
    n_vector_cnt = len(eyes_vectors['hhhh'])  # equal to pre-defined SVMS_PER_EACH_PROPERTY value

    ohlora_z_vectors = load_ohlora_z_vectors_v8(vector_csv_path=OHLORA_Z_VECTOR_CSV_PATH_V8)
    ohlora_w_group_names = load_ohlora_w_group_names_v8(group_name_csv_path=OHLORA_W_GROUP_NAME_CSV_PATH_V8)

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

                generate_ombre_img(vectorfind_v8_generator, hair_seg_model, eyes_vector, mouth_vector, pose_vector,
                                   code_w, save_dir, img_file_name, pms,
                                   color=(pm_idx / pm_cnt), ombre_height=0.4, ombre_grad_height=0.6)


# StyleGAN-VectorFind-v8 옴브레 염색 적용 이미지 생성 테스트 (모델 로딩을 포함한 전 과정)
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - hair_seg_model (nn.Module) : Hair 영역 추출용 Segmentation Model

def generate_ombre_image_using_v8_all_process(hair_seg_model):
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
                                  hair_seg_model,
                                  eyes_vectors,
                                  mouth_vectors,
                                  pose_vectors)
