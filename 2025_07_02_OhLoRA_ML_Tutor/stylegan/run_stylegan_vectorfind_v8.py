
import torch
import pandas as pd
import numpy as np
import os

try:
    from stylegan_common.visualizer import postprocess_image
except:
    from stylegan.stylegan_common.visualizer import postprocess_image


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
GROUP_NAMES = ['hhhh', 'hhhl', 'hhlh', 'hhll', 'hlhh', 'hlhl', 'hllh', 'hlll',
               'lhhh', 'lhhl', 'lhlh', 'lhll', 'llhh', 'llhl', 'lllh', 'llll']
GROUP_NAMES_V7 = ['hhh', 'hhl', 'hlh', 'hll', 'lhh', 'lhl', 'llh', 'lll']


def generate_image_using_w(finetune_v8_generator, w, trunc_psi=1.0, trunc_layers=0, randomize_noise=False, lod=None):
    with torch.no_grad():
        wp = finetune_v8_generator.truncation(w, trunc_psi, trunc_layers)
        images = finetune_v8_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
        images = postprocess_image(images.detach().cpu().numpy())
    return images


# Oh-LoRA 이미지 생성용 latent z vector 가 저장된 파일을 먼저 불러오기 시도
# Create Date : 2025.08.01
# Last Update Date : -

# Arguments:
# - vector_csv_path (str) : latent z vector 가 저장된 csv 파일의 경로

# Returns:
# - ohlora_z_vectors (NumPy array or None) : Oh-LoRA 이미지 생성용 latent z vector (불러오기 성공 시)
#                                            None (불러오기 실패 시)

def load_ohlora_z_vectors(vector_csv_path):
    ohlora_z_vectors_df = pd.read_csv(vector_csv_path)
    ohlora_z_vectors = np.array(ohlora_z_vectors_df)
    print(f'Oh-LoRA z vector (StyleGAN-VectorFind-v8) load successful!! 👱‍♀️✨')

    return ohlora_z_vectors


# Oh-LoRA 이미지 생성용 intermediate w vector 각각에 대해, group name 정보를 먼저 불러오기 시도
# Create Date : 2025.08.01
# Last Update Date : -

# Arguments:
# - group_name_csv_path (str) : intermediate w vector 에 대한 group name 정보가 저장된 csv 파일의 경로

# Returns:
# - group_names (list(str) or None) : Oh-LoRA 이미지 생성용 intermediate w vector 에 대한 group name 의 list (불러오기 성공 시)
#                                     None (불러오기 실패 시)

def load_ohlora_w_group_names(group_name_csv_path):
    ohlora_w_vectors_df = pd.read_csv(group_name_csv_path)
    group_names = ohlora_w_vectors_df['group_name'].tolist()
    print(f'group names for each Oh-LoRA w vector (StyleGAN-VectorFind-v8) load successful!! 👱‍♀️✨')

    return group_names


# Property Score 값을 변경하기 위해 intermediate w vector 에 가감할 벡터 정보 반환 ('hhhh', 'hhhl', ..., 'llll' 의 각 그룹 별)
# Create Date : 2025.08.01
# Last Update Date : -

# Arguments:
# - vectorfind_version (str) : Oh-LoRA latent z vector & w vector 를 위한 StyleGAN-VectorFind 버전 ('v7' or 'v8')

# Returns:
# - eyes_vectors  (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors  (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def get_property_change_vectors(vectorfind_version):
    vector_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_{vectorfind_version}/property_score_vectors'

    if vectorfind_version == 'v7':
        group_names_list = GROUP_NAMES_V7
    else:  # v8
        group_names_list = GROUP_NAMES

    eyes_vectors = {}
    mouth_vectors = {}
    pose_vectors = {}

    for group_name in group_names_list:
        eyes_vector = np.array(pd.read_csv(f'{vector_save_dir}/eyes_change_w_vector_{group_name}.csv',
                                           index_col=0))

        mouth_vector = np.array(pd.read_csv(f'{vector_save_dir}/mouth_change_w_vector_{group_name}.csv',
                                            index_col=0))

        pose_vector = np.array(pd.read_csv(f'{vector_save_dir}/pose_change_w_vector_{group_name}.csv',
                                           index_col=0))

        eyes_vectors[group_name] = eyes_vector
        mouth_vectors[group_name] = mouth_vector
        pose_vectors[group_name] = pose_vector

    return eyes_vectors, mouth_vectors, pose_vectors
