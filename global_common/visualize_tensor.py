import numpy as np
import cv2


# PyTorch Tensor 를 이미지로 저장
# Create Date : 2025.04.13
# Last Update Date : -

# args :
# - image_tensor    (Tensor) : 이미지로 저장할 PyTorch Tensor (dimension: (3, H, W))
# - image_save_path (str)    : 이미지로 저장할 경로
# - max_val         (float)  : 이미지 픽셀의 최댓값
# - min_val         (float)  : 이미지 픽셀의 최솟값

# returns :
# - val_accuracy (float) : 모델의 validation 정확도
# - val_loss     (float) : 모델의 validation loss

def save_tensor_png(image_tensor, image_save_path, max_val=1, min_val=-1):
    img_ = np.array(image_tensor.detach().cpu())
    img_ = np.transpose(img_, (1, 2, 0))
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_ = (img_ - min_val) * 255 / (max_val - min_val)

    result, overlay_image_arr = cv2.imencode(ext='.png',
                                             img=img_,
                                             params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

    if result:
        with open(image_save_path, mode='w+b') as f:
            overlay_image_arr.tofile(f)