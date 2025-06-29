
from generate_ohlora_image import generate_images
import cv2


# Oh-LoRA (오로라) 이미지 생성 및 표시
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - vectorfind_generator (nn.Module)   : StyleGAN-VectorFind-v7 또는 StyleGAN-VectorFind-v8 의 Generator
# - ohlora_z_vector      (NumPy array) : Oh-LoRA 이미지 생성용 latent z vector, dim = (512 + 3,) (v7) or (512 + 7,) (v8)
# - eyes_vector          (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - mouth_vector         (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - pose_vector          (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터, dim = (512,)
# - eyes_score           (float)       : 눈을 뜬 정도 (eyes) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)
# - mouth_score          (float)       : 입을 벌린 정도 (mouth) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)
# - pose_score           (float)       : 고개 돌림 (pose) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)

# Returns :
# - 직접 반환되는 값 없음
# - final_product/ohlora.png 경로에 오로라 이미지 생성 및 화면에 display

def generate_and_show_ohlora_image(vectorfind_generator, ohlora_z_vector, eyes_vector, mouth_vector, pose_vector,
                                   eyes_score, mouth_score, pose_score):

    ohlora_image_to_display = generate_images(vectorfind_generator, ohlora_z_vector,
                                              eyes_vector, mouth_vector, pose_vector,
                                              eyes_pm=eyes_score, mouth_pm=mouth_score, pose_pm=pose_score)

    cv2.imshow('Oh-LoRA', ohlora_image_to_display[:, :, ::-1])
    _ = cv2.waitKey(1)
