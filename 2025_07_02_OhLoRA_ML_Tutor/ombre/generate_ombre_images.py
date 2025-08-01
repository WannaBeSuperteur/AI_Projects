
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import os

from stylegan.stylegan_common.visualizer import postprocess_image, save_image

IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_W = 512

seg_model_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


def generate_image_using_w(vectorfind_generator, w, trunc_psi=1.0, trunc_layers=0, randomize_noise=False, lod=None):
    with torch.no_grad():
        wp = vectorfind_generator.truncation(w, trunc_psi, trunc_layers)
        images = vectorfind_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
        images = postprocess_image(images.detach().cpu().numpy())
    return images


# 옴브레 스타일 적용
# Create Date : 2025.08.01
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


# 옴브레 염색 적용된 이미지 생성
# Create Date : 2025.08.01
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
