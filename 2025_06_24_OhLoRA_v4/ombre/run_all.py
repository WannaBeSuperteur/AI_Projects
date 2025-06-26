
from generate_ombre_images import generate_ombre_image_using_v7_all_process, generate_ombre_image_using_v8_all_process
from generate_gif import generate_gif
from generate_opencv_screen import create_opencv_screen

import torch

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from segmentation.seg_model_ohlora_v4.model import SegModelForOhLoRAV4


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

    v7_test_ohlora_nos = [672, 1277, 1836, 1918, 2137]
    v8_test_ohlora_nos = [83, 194, 1180, 1313, 1996]

    # GIF 생성 테스트
    for v7_test_ohlora_no in v7_test_ohlora_nos:
        generate_gif(vectorfind_ver='v7', ohlora_no=v7_test_ohlora_no)

    for v8_test_ohlora_no in v8_test_ohlora_nos:
        generate_gif(vectorfind_ver='v8', ohlora_no=v8_test_ohlora_no)

    # OpenCV 움직이는 화면 생성 테스트
    for v7_test_ohlora_no in v7_test_ohlora_nos:
        create_opencv_screen(vectorfind_ver='v7', ohlora_no=v7_test_ohlora_no)

    for v8_test_ohlora_no in v8_test_ohlora_nos:
        create_opencv_screen(vectorfind_ver='v8', ohlora_no=v8_test_ohlora_no)
