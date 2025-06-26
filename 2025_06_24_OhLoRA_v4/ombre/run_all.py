
from generate_ombre_images import generate_ombre_image_using_v7_all_process, generate_ombre_image_using_v8_all_process
from generate_gif import generate_gif
from generate_opencv_screen import create_opencv_screen

import torch

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from segmentation.seg_model_ohlora_v4.model import SegModelForOhLoRAV4
from stylegan.common import load_existing_stylegan_vectorfind_v7, load_existing_stylegan_vectorfind_v8
import stylegan.stylegan_common.stylegan_generator as gen

IMAGE_RESOLUTION = 256


# Oh-LoRA v4 용 Hair 영역 추출 Segmentation 모델 로딩
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - hair_seg_model (OrderedDict) : Oh-LoRA v4 용 Hair 영역 추출 Segmentation 모델

def load_existing_hair_seg_model(device):
    hair_seg_model = SegModelForOhLoRAV4()

    # load generator state dict
    hair_seg_model_path = f'{PROJECT_DIR_PATH}/segmentation/models/segmentation_model_ohlora_v4.pth'
    hair_seg_model_state_dict = torch.load(hair_seg_model_path, map_location=device, weights_only=True)
    hair_seg_model.load_state_dict(hair_seg_model_state_dict)

    hair_seg_model.to(device)
    return hair_seg_model


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training : {device}')

    # load segmentation model
    hair_seg_model = load_existing_hair_seg_model(device)
    generate_ombre_image_using_v7_all_process(hair_seg_model)
    generate_ombre_image_using_v8_all_process(hair_seg_model)

    # load StyleGAN-VectorFind-v7 and v8
    stylegan_vectorfind_v7_generator = gen.StyleGANGeneratorForV6(resolution=IMAGE_RESOLUTION)
    stylegan_vectorfind_v7_state_dict = load_existing_stylegan_vectorfind_v7(device)
    stylegan_vectorfind_v7_generator.load_state_dict(stylegan_vectorfind_v7_state_dict)
    stylegan_vectorfind_v7_generator.to(device)

    stylegan_vectorfind_v8_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)
    stylegan_vectorfind_v8_state_dict = load_existing_stylegan_vectorfind_v8(device)
    stylegan_vectorfind_v8_generator.load_state_dict(stylegan_vectorfind_v8_state_dict)
    stylegan_vectorfind_v8_generator.to(device)

    v7_test_ohlora_nos = [672, 1277, 1836, 1918, 2137]
    v8_test_ohlora_nos = [83, 194, 1180, 1313, 1996]

    # GIF 생성 테스트
    color_list = [0.025 * x for x in range(40)]
    ombre_height_list = [0.1 + 0.01 * x for x in range(40)]
    ombre_grad_height_list = [0.05 + 0.015 * x for x in range(40)]
    pms_list = {
        'eyes': [-1.2 + 0.12 * x for x in range(20)] + [1.2 - 0.12 * x for x in range(20)],
        'mouth': [1.2 - 0.06 * x for x in range(40)],
        'pose': [0.6 - 0.09 * x for x in range(20)] + [-1.2 for _ in range(20)]
    }

    gif_save_dir_path = f'{PROJECT_DIR_PATH}/ombre/test_gif'
    os.makedirs(f'{gif_save_dir_path}/v7', exist_ok=True)
    os.makedirs(f'{gif_save_dir_path}/v8', exist_ok=True)

    for v7_test_ohlora_no in v7_test_ohlora_nos:
        generate_gif(vectorfind_generator=stylegan_vectorfind_v7_generator,
                     hair_seg_model=hair_seg_model,
                     vectorfind_ver='v7',
                     ohlora_no=v7_test_ohlora_no,
                     color_list=color_list,
                     ombre_height_list=ombre_height_list,
                     ombre_grad_height_list=ombre_grad_height_list,
                     pms_list=pms_list,
                     gif_save_path=f'{gif_save_dir_path}/v7/{v7_test_ohlora_no}.gif',
                     duration=0.05)

    for v8_test_ohlora_no in v8_test_ohlora_nos:
        generate_gif(vectorfind_generator=stylegan_vectorfind_v8_generator,
                     hair_seg_model=hair_seg_model,
                     vectorfind_ver='v8',
                     ohlora_no=v8_test_ohlora_no,
                     color_list=color_list,
                     ombre_height_list=ombre_height_list,
                     ombre_grad_height_list=ombre_grad_height_list,
                     pms_list=pms_list,
                     gif_save_path=f'{gif_save_dir_path}/v8/{v8_test_ohlora_no}.gif',
                     duration=0.05)

    # OpenCV 움직이는 화면 생성 테스트
    color_list_opencv_test = [0.0125 * x for x in range(80)]
    ombre_height_list_opencv_test = [0.1 + 0.005 * x for x in range(80)]
    ombre_grad_height_list_opencv_test = [0.05 + 0.0075 * x for x in range(80)]
    pms_list_opencv_test = {
        'eyes': [-1.2 + 0.06 * x for x in range(40)] + [1.2 - 0.06 * x for x in range(40)],
        'mouth': [1.2 - 0.03 * x for x in range(80)],
        'pose': [0.6 - 0.045 * x for x in range(40)] + [-1.2 for _ in range(40)]
    }

    for v7_test_ohlora_no in v7_test_ohlora_nos:
        create_opencv_screen(vectorfind_generator=stylegan_vectorfind_v7_generator,
                             hair_seg_model=hair_seg_model,
                             vectorfind_ver='v7',
                             ohlora_no=v7_test_ohlora_no,
                             color_list=color_list_opencv_test,
                             ombre_height_list=ombre_height_list_opencv_test,
                             ombre_grad_height_list=ombre_grad_height_list_opencv_test,
                             pms_list=pms_list_opencv_test)

    for v8_test_ohlora_no in v8_test_ohlora_nos:
        create_opencv_screen(vectorfind_generator=stylegan_vectorfind_v8_generator,
                             hair_seg_model=hair_seg_model,
                             vectorfind_ver='v8',
                             ohlora_no=v8_test_ohlora_no,
                             color_list=color_list_opencv_test,
                             ombre_height_list=ombre_height_list_opencv_test,
                             ombre_grad_height_list=ombre_grad_height_list_opencv_test,
                             pms_list=pms_list_opencv_test)
