
import torch
import pandas as pd
import numpy as np

try:
    from stylegan_common.visualizer import postprocess_image
except:
    from stylegan.stylegan_common.visualizer import postprocess_image


ORIGINAL_HIDDEN_DIMS_W = 512


def generate_image_using_mid_vector(finetune_v9_generator, mid_vector, layer_name,
                                    trunc_psi=1.0, trunc_layers=0, randomize_noise=False, lod=None):

    if layer_name == 'w':
        with torch.no_grad():
            wp = finetune_v9_generator.truncation(mid_vector, trunc_psi, trunc_layers)
            images = finetune_v9_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
            images = postprocess_image(images.detach().cpu().numpy())

    elif layer_name == 'mapping_split1':
        with torch.no_grad():
            w1 = mid_vector[:, :ORIGINAL_HIDDEN_DIMS_W]
            w2 = mid_vector[:, ORIGINAL_HIDDEN_DIMS_W:]
            w1_ = finetune_v9_generator.mapping.dense7(w1.cuda()).detach().cpu()
            w2_ = finetune_v9_generator.mapping.dense_new1(w2.cuda()).detach().cpu()
            w = w1_ + w2_

            wp = finetune_v9_generator.truncation(w, trunc_psi, trunc_layers)
            images = finetune_v9_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
            images = postprocess_image(images.detach().cpu().numpy())

    else:  # mapping_split2
        with torch.no_grad():
            w1_ = mid_vector[:, :ORIGINAL_HIDDEN_DIMS_W]
            w2_ = mid_vector[:, ORIGINAL_HIDDEN_DIMS_W:]
            w = w1_ + w2_

            wp = finetune_v9_generator.truncation(w, trunc_psi, trunc_layers)
            images = finetune_v9_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
            images = postprocess_image(images.detach().cpu().numpy())

    return images


# 이미지 생성을 위한 concatenated intermediate vector 생성 및 반환
# Create Date : 2025.06.10
# Last Update Date : -

# Arguments:
# - finetune_v9_generator (nn.Module) : StyleGAN-FineTune-v9 의 Generator
# - layer_name            (str)       : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름
#                                       ('mapping_split1', 'mapping_split2' or 'w')
# - code_part1            (Tensor)    : latent z vector 의 앞부분 (dim = 512)
# - code_part2            (Tensor)    : latent z vector 의 뒷부분 (dim = 7)

# Returns:
# - code_mid (Tensor) : 이미지 생성을 위한 concatenated intermediate vector

def generate_code_mid(finetune_v9_generator, layer_name, code_part1, code_part2):
    with torch.no_grad():
        if layer_name == 'w':
            code_mid = finetune_v9_generator.mapping(code_part1.cuda(), code_part2.cuda())['w'].detach().cpu()

        elif layer_name == 'mapping_split1':
            code_w1 = finetune_v9_generator.mapping(code_part1.cuda(), code_part2.cuda())['w1'].detach().cpu()
            code_w2 = finetune_v9_generator.mapping(code_part1.cuda(), code_part2.cuda())['w2'].detach().cpu()
            code_mid = torch.concat([code_w1, code_w2], dim=1)

        else:  # mapping_split2
            code_w1_ = finetune_v9_generator.mapping(code_part1.cuda(), code_part2.cuda())['w1_'].detach().cpu()
            code_w2_ = finetune_v9_generator.mapping(code_part1.cuda(), code_part2.cuda())['w2_'].detach().cpu()
            code_mid = torch.concat([code_w1_, code_w2_], dim=1)

    return code_mid


# Oh-LoRA 이미지 생성용 latent z vector 가 저장된 파일을 먼저 불러오기 시도
# Create Date : 2025.06.10
# Last Update Date : -

# Arguments:
# - vector_csv_path (str) : latent z vector 가 저장된 csv 파일의 경로

# Returns:
# - ohlora_z_vectors (NumPy array or None) : Oh-LoRA 이미지 생성용 latent z vector (불러오기 성공 시)
#                                            None (불러오기 실패 시)

def load_ohlora_z_vectors(vector_csv_path):
    try:
        ohlora_z_vectors_df = pd.read_csv(vector_csv_path)
        ohlora_z_vectors = np.array(ohlora_z_vectors_df)
        print(f'Oh-LoRA z vector load successful!! 👱‍♀️✨')
        return ohlora_z_vectors

    except Exception as e:
        print(f'Oh-LoRA z vector load failed ({e}), using random-generated z vectors')
        return None


# 이미지 50장 생성 후 비교 테스트를 위한, property score label (intermediate vector 에 n vector 를 가감할 때의 가중치) 생성 및 반환
# Create Date : 2025.06.10
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - eyes_pm_order  (list(float)) : eyes (눈을 뜬 정도) 속성에 대한 50장 각각의 property score label
# - mouth_pm_order (list(float)) : mouth (입을 벌린 정도) 속성에 대한 50장 각각의 property score label
# - pose_pm_order  (list(float)) : pose (고개 돌림) 속성에 대한 50장 각각의 property score label

def get_pm_labels():
    eyes_pms = [-1.2, 1.2]
    mouth_pms = [-1.8, -0.9, 0.0, 0.9, 1.8]
    pose_pms = [-1.8, -1.2, -0.6, 0.0, 0.6]

    eyes_pm_order = []
    mouth_pm_order = []
    pose_pm_order = []

    for mouth in mouth_pms:
        for eyes in eyes_pms:
            for pose in pose_pms:
                eyes_pm_order.append(eyes)
                mouth_pm_order.append(mouth)
                pose_pm_order.append(pose)

    return eyes_pm_order, mouth_pm_order, pose_pm_order
