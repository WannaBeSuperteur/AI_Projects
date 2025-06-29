
import torch
from stylegan.run_stylegan_vectorfind_v8 import generate_image_using_w


ORIGINAL_HIDDEN_DIMS_Z = 512
kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)


# Oh-LoRA 👱‍♀️ (오로라) 이미지 생성 및 반환
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - vectorfind_generator (nn.Module)   : StyleGAN-VectorFind-v7 또는 StyleGAN-VectorFind-v8 의 Generator
# - ohlora_z_vector      (NumPy array) : Oh-LoRA 이미지 생성용 latent z vector, dim = (512 + 3,) (v7) or (512 + 7,) (v8)
# - eyes_vector          (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - mouth_vector         (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - pose_vector          (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터, dim = (512,)
# - eyes_pm              (float)       : ohlora_z_vector 에 eyes 핵심 속성 값 변화 벡터를 더하는 가중치
# - mouth_pm             (float)       : ohlora_z_vector 에 mouth 핵심 속성 값 변화 벡터를 더하는 가중치
# - pose_pm              (float)       : ohlora_z_vector 에 pose 핵심 속성 값 변화 벡터를 더하는 가중치

# Returns:
# - ohlora_image (NumPy array) : 생성된 이미지

def generate_images(vectorfind_generator, ohlora_z_vector, eyes_vector, mouth_vector, pose_vector,
                    eyes_pm, mouth_pm, pose_pm):

    code_part1_np = ohlora_z_vector[:ORIGINAL_HIDDEN_DIMS_Z]
    code_part2_np = ohlora_z_vector[ORIGINAL_HIDDEN_DIMS_Z:]
    code_part1 = torch.tensor(code_part1_np).unsqueeze(0).to(torch.float32)  # 512
    code_part2 = torch.tensor(code_part2_np).unsqueeze(0).to(torch.float32)  # 3 (v7) or 7 (v8)

    # generate image
    with torch.no_grad():
        code_w = vectorfind_generator.mapping(code_part1.cuda(), code_part2.cuda())['w'].detach().cpu()

        code_w_ = code_w + eyes_pm * torch.tensor(eyes_vector[:ORIGINAL_HIDDEN_DIMS_Z])
        code_w_ = code_w_ + mouth_pm * torch.tensor(mouth_vector[:ORIGINAL_HIDDEN_DIMS_Z])
        code_w_ = code_w_ + pose_pm * torch.tensor(pose_vector[:ORIGINAL_HIDDEN_DIMS_Z])
        code_w_ = code_w_.type(torch.float32)

        images = generate_image_using_w(vectorfind_generator, code_w_)

    ohlora_image = images[0]
    return ohlora_image
