
import torch
from stylegan.stylegan_common.visualizer import postprocess_image


ORIGINAL_HIDDEN_DIMS_Z = 512
kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)


# Oh-LoRA 👱‍♀️ (오로라) 이미지 생성 및 반환
# Create Date : 2025.05.15
# Last Update Date : -

# Arguments:
# - vectorfind_v6_generator (nn.Module)   : StyleGAN-VectorFind-v6 의 Generator
# - ohlora_z_vector         (NumPy array) : Oh-LoRA 이미지 생성용 latent z vector, dim = (512 + 3,)
# - eyes_vector             (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터, dim = (512 + 3,)
# - mouth_vector            (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터, dim = (512 + 3,)
# - pose_vector             (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터, dim = (512 + 3,)
# - eyes_pm                 (float)       : ohlora_z_vector 에 eyes 핵심 속성 값 변화 벡터를 더하는 가중치
# - mouth_pm                (float)       : ohlora_z_vector 에 mouth 핵심 속성 값 변화 벡터를 더하는 가중치
# - pose_pm                 (float)       : ohlora_z_vector 에 pose 핵심 속성 값 변화 벡터를 더하는 가중치

# Returns:
# - ohlora_image (NumPy array) : 생성된 이미지

def generate_images(vectorfind_v6_generator, ohlora_z_vector, eyes_vector, mouth_vector, pose_vector,
                    eyes_pm, mouth_pm, pose_pm):

    code_part1_np = ohlora_z_vector[:ORIGINAL_HIDDEN_DIMS_Z]
    code_part2_np = ohlora_z_vector[ORIGINAL_HIDDEN_DIMS_Z:]
    code_part1 = torch.tensor(code_part1_np).unsqueeze(0).to(torch.float32)  # 512
    code_part2 = torch.tensor(code_part2_np).unsqueeze(0).to(torch.float32)  # 3

    # generate image
    with torch.no_grad():
        code_part1_ = code_part1 + eyes_pm * torch.tensor(eyes_vector[:ORIGINAL_HIDDEN_DIMS_Z])
        code_part1_ = code_part1_ + mouth_pm * torch.tensor(mouth_vector[:ORIGINAL_HIDDEN_DIMS_Z])
        code_part1_ = code_part1_ + pose_pm * torch.tensor(pose_vector[:ORIGINAL_HIDDEN_DIMS_Z])
        code_part1_ = code_part1_.type(torch.float32)

        code_part2_ = code_part2 + eyes_pm * torch.tensor(eyes_vector[ORIGINAL_HIDDEN_DIMS_Z:])
        code_part2_ = code_part2_ + mouth_pm * torch.tensor(mouth_vector[ORIGINAL_HIDDEN_DIMS_Z:])
        code_part2_ = code_part2_ + pose_pm * torch.tensor(pose_vector[ORIGINAL_HIDDEN_DIMS_Z:])
        code_part2_ = code_part2_.type(torch.float32)

        images = vectorfind_v6_generator(code_part1_.cuda(), code_part2_.cuda(), **kwargs_val)['image']
        images = postprocess_image(images.detach().cpu().numpy())

    ohlora_image = images[0]
    return ohlora_image
