from generate_ombre_images import generate_ombre_image
from stylegan.run_stylegan_vectorfind_v7 import get_property_change_vectors as get_property_change_vectors_v7
from stylegan.run_stylegan_vectorfind_v8 import get_property_change_vectors as get_property_change_vectors_v8

import imageio


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
                 color_list, ombre_height_list, ombre_grad_height_list, pms_list, gif_save_path, duration=0.1):

    print(f'generating GIF image for {vectorfind_ver} / {ohlora_no} ...')
    eyes_vectors, mouth_vectors, pose_vectors = get_property_score_change_vectors(vectorfind_ver)
    ohlora_images = []
    n_frames = len(color_list)

    for idx in range(n_frames):
        color = color_list[idx]
        ombre_height = ombre_height_list[idx]
        ombre_grad_height = ombre_grad_height_list[idx]
        pms = {'eyes': pms_list['eyes'][idx], 'mouth': pms_list['mouth'][idx], 'pose': pms_list['pose'][idx]}

        ohlora_image = generate_ombre_image(vectorfind_generator, hair_seg_model,
                                            eyes_vectors, mouth_vectors, pose_vectors,
                                            vectorfind_ver, ohlora_no, color, ombre_height, ombre_grad_height, pms)
        ohlora_images.append(ohlora_image)

    imageio.mimsave(gif_save_path, ohlora_images, duration=duration)


# GIF 이미지 생성
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_ver (str) : StyleGAN-VectorFind 버전 ('v7' or 'v8')

# Returns:
# - eyes_vectors  (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors  (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def get_property_score_change_vectors(vectorfind_ver):
    if vectorfind_ver == 'v7':
        eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors_v7()
    else:  # v8
        eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors_v8(vectorfind_ver)

    return eyes_vectors, mouth_vectors, pose_vectors

