from torchvision.io import read_image

try:
    from stylegan_common.visualizer import postprocess_image, save_image
    import stylegan_common.stylegan_generator as gen

    from common import (load_existing_stylegan_finetune_v8,
                        load_existing_stylegan_vectorfind_v8,
                        stylegan_transform,
                        load_merged_property_score_cnn)

except:
    from stylegan.stylegan_common.visualizer import postprocess_image, save_image
    import stylegan.stylegan_common.stylegan_generator as gen

    from stylegan.common import (load_existing_stylegan_finetune_v8,
                                 load_existing_stylegan_vectorfind_v8,
                                 stylegan_transform,
                                 load_merged_property_score_cnn)

import torch
import os
import numpy as np
import pandas as pd

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256

ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINAL_HIDDEN_DIMS_W = 512
ORIGINALLY_PROPERTY_DIMS = 7    # 원래 property (eyes, hair_color, hair_length, mouth, pose,
                                #               background_mean, background_std) 목적으로 사용된 dimension 값

TEST_IMG_CASES = 1

OHLORA_FINAL_VECTORS_TEST_REPORT_PATH = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/final_vector_test_report'
os.makedirs(OHLORA_FINAL_VECTORS_TEST_REPORT_PATH, exist_ok=True)

GROUP_NAMES = ['hhhh', 'hhhl', 'hhlh', 'hhll', 'hlhh', 'hlhl', 'hllh', 'hlll',
               'lhhh', 'lhhl', 'lhlh', 'lhll', 'llhh', 'llhl', 'lllh', 'llll']
GROUP_NAMES_V7 = ['hhh', 'hhl', 'hlh', 'hll', 'lhh', 'lhl', 'llh', 'lll']
PROPERTY_NAMES = ['eyes', 'mouth', 'pose']

kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)


def generate_image_using_w(finetune_v8_generator, w, trunc_psi=1.0, trunc_layers=0, randomize_noise=False, lod=None):
    with torch.no_grad():
        wp = finetune_v8_generator.truncation(w, trunc_psi, trunc_layers)
        images = finetune_v8_generator.synthesis(wp.cuda(), lod, randomize_noise)['image']
        images = postprocess_image(images.detach().cpu().numpy())
    return images


# Property Score 값을 변경하기 위해 intermediate w vector 에 가감할 벡터 정보 반환 ('hhhh', 'hhhl', ..., 'llll' 의 각 그룹 별)
# Create Date : 2025.06.24
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


# Oh-LoRA 이미지 생성용 latent z vector 가 저장된 파일을 먼저 불러오기 시도
# Create Date : 2025.06.24
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
# Create Date : 2025.06.24
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


# 이미지 50장 생성 후 의도한 property score label 과, 생성된 이미지에 대한 CNN 예측 property score 를 비교 테스트 (corr-coef)
# Create Date : 2025.06.24
# Last Update Date : -

# Arguments:
# - finetune_v8_generator (nn.Module)         : StyleGAN-FineTune-v8 의 Generator
# - property_score_cnn    (nn.Module)         : 핵심 속성 값을 계산하기 위한 CNN
# - eyes_vectors          (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors         (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors          (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

# Returns:
# - stylegan_vectorfind_v8/inference_test_after_training 디렉토리에 이미지 생성
# - stylegan_vectorfind_v8/image_generation_report 디렉토리에 테스트 결과를 csv 파일로 저장

def run_property_score_compare_test(finetune_v8_generator, property_score_cnn, eyes_vectors, mouth_vectors,
                                    pose_vectors):

    n_vector_cnt = len(eyes_vectors['hhhh'])  # equal to pre-defined SVMS_PER_EACH_PROPERTY value
    passed_count = 0

    ohlora_z_vector_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_z_vectors.csv'
    ohlora_w_group_name_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/ohlora_w_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_w_group_names = load_ohlora_w_group_names(group_name_csv_path=ohlora_w_group_name_csv_path)

    # label: 'eyes', 'mouth', 'pose'
    eyes_pm_order, mouth_pm_order, pose_pm_order = get_pm_labels()
    pm_cnt = len(eyes_pm_order)

    all_data_dict = {'case': [], 'vector_no': [], 'passed': [], 'group_name': [],
                     'eyes_corr': [], 'mouth_corr': [], 'pose_corr': []}

    count_to_generate = len(ohlora_z_vectors)
    code_part1s_np = np.zeros((count_to_generate, ORIGINAL_HIDDEN_DIMS_Z))
    code_part2s_np = np.zeros((count_to_generate, ORIGINALLY_PROPERTY_DIMS))
    generated_count = 0

    # image generation
    for i in range(count_to_generate):
        save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/inference_test_after_training/test_{i:04d}'
        os.makedirs(save_dir, exist_ok=True)

        code_part1s_np[i] = ohlora_z_vectors[i][:ORIGINAL_HIDDEN_DIMS_Z]
        code_part2s_np[i] = ohlora_z_vectors[i][ORIGINAL_HIDDEN_DIMS_Z:]
        code_part1 = torch.tensor(code_part1s_np[i]).unsqueeze(0).to(torch.float32)  # 512
        code_part2 = torch.tensor(code_part2s_np[i]).unsqueeze(0).to(torch.float32)  # 7

        with torch.no_grad():
            code_w = finetune_v8_generator.mapping(code_part1.cuda(), code_part2.cuda())['w'].detach().cpu()

        for vi in range(n_vector_cnt):
            n_vector_idx = i * n_vector_cnt + vi
            group_name = ohlora_w_group_names[n_vector_idx]

            eyes_vector = eyes_vectors[group_name]
            mouth_vector = mouth_vectors[group_name]
            pose_vector = pose_vectors[group_name]

            eyes_scores, mouth_scores, pose_scores = [], [], []

            all_data_dict['case'].append(i)
            all_data_dict['vector_no'].append(vi)
            all_data_dict['group_name'].append(group_name)

            for pm_idx in range(pm_cnt):
                img_file_name = f'case_{i:03d}_{vi:03d}_pm_{pm_idx:03d}.jpg'
                pms = {'eyes': eyes_pm_order[pm_idx], 'mouth': mouth_pm_order[pm_idx], 'pose': pose_pm_order[pm_idx]}

                generate_image(finetune_v8_generator, property_score_cnn, eyes_vector, mouth_vector, pose_vector,
                               eyes_scores, mouth_scores, pose_scores, code_w, save_dir, img_file_name, vi, pms)

            # compute and record corr-coef
            eyes_corrcoef = np.corrcoef(eyes_pm_order, eyes_scores)[0][1]
            mouth_corrcoef = np.corrcoef(mouth_pm_order, mouth_scores)[0][1]
            pose_corrcoef = np.corrcoef(pose_pm_order, pose_scores)[0][1]

            all_data_dict['eyes_corr'].append(round(eyes_corrcoef, 4))
            all_data_dict['mouth_corr'].append(round(mouth_corrcoef, 4))
            all_data_dict['pose_corr'].append(round(pose_corrcoef, 4))

            # check passed
            generated_count += 1

            passed = abs(eyes_corrcoef) >= 0.92 and abs(mouth_corrcoef) >= 0.88 and abs(pose_corrcoef) >= 0.88
            eyes_diff = max(0.92 - abs(eyes_corrcoef), 0)
            mouth_diff = max(0.88 - abs(mouth_corrcoef), 0)
            pose_diff = max(0.88 - abs(pose_corrcoef), 0)

            pass_diff = eyes_diff + mouth_diff + pose_diff
            diff = {'eyes': round(eyes_diff, 4), 'mouth': round(mouth_diff, 4), 'pose': round(pose_diff, 4)}

            if passed:
                passed_count += 1

            passed = 'O' if passed else 'X'
            all_data_dict['passed'].append(passed)

            # save data for case
            case_data_dict = {'pm_idx': list(range(pm_cnt)),
                              'eyes_pm': eyes_pm_order, 'mouth_pm': mouth_pm_order, 'pose_pm': pose_pm_order,
                              'eyes_score': eyes_scores, 'mouth_score': mouth_scores, 'pose_score': pose_scores}
            case_data_df = pd.DataFrame(case_data_dict)

            case_data_save_path = f'{save_dir}/case_{i:03d}_{vi:03d}_result.csv'
            case_data_df.to_csv(case_data_save_path, index=False)

            print(f'testing idx {i} vector {vi} ... (passed : {passed_count}, current total gap: {round(pass_diff, 4)}, '
                  f'diff: {diff})')

    print('Already loaded "saved z vectors info" for Oh-LoRA face image generation.')
    image_gen_report_path = OHLORA_FINAL_VECTORS_TEST_REPORT_PATH

    # save all data
    all_data_df = pd.DataFrame(all_data_dict)
    all_data_df['sum_abs_corr'] = abs(all_data_df['eyes_corr']) + abs(all_data_df['mouth_corr']) + abs(all_data_df['pose_corr'])
    all_data_df['sum_abs_corr'] = all_data_df['sum_abs_corr'].apply(lambda x: round(x, 4))

    all_data_save_path = f'{image_gen_report_path}/test_result.csv'
    all_data_df.to_csv(all_data_save_path, index=False)

    # compute statistics
    eyes_corr_mean = all_data_df['eyes_corr'].mean()
    mouth_corr_mean = all_data_df['mouth_corr'].mean()
    pose_corr_mean = all_data_df['pose_corr'].mean()
    sum_abs_corr_mean = all_data_df['sum_abs_corr'].mean()

    statistics_df = pd.DataFrame({'eyes_corr_mean': [round(eyes_corr_mean, 4)],
                                  'mouth_corr_mean': [round(mouth_corr_mean, 4)],
                                  'pose_corr_mean': [round(pose_corr_mean, 4)],
                                  'sum_abs_corr_mean': [round(sum_abs_corr_mean, 4)],
                                  'passed': passed_count,
                                  'passed_ratio': passed_count / (generated_count * n_vector_cnt)})

    statistics_save_path = f'{image_gen_report_path}/test_statistics.csv'
    statistics_df.to_csv(statistics_save_path)

    # save latent codes (w)
    code_part1s_np = np.round(code_part1s_np[:generated_count], 4)
    code_part2s_np = np.round(code_part2s_np[:generated_count], 4)
    code_all_np = np.concatenate([code_part1s_np, code_part2s_np], axis=1)

    pd.DataFrame(code_part1s_np).to_csv(f'{image_gen_report_path}/latent_codes_part1.csv', index=False)
    pd.DataFrame(code_part2s_np).to_csv(f'{image_gen_report_path}/latent_codes_part2.csv', index=False)
    pd.DataFrame(code_all_np).to_csv(f'{image_gen_report_path}/latent_codes_all.csv', index=False)


# 주어진 eyes, mouth, pose 핵심 속성 값 변화 벡터를 이용하여 이미지 생성
# Create Date : 2025.06.24
# Last Update Date : -

# Arguments:
# - finetune_v8_generator (nn.Module)   : StyleGAN-FineTune-v8 의 Generator
# - property_score_cnn    (nn.Module)   : 핵심 속성 값을 계산하기 위한 CNN
# - eyes_vector           (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터
# - mouth_vector          (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터
# - pose_vector           (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터
# - eyes_scores           (list)        : Property Score CNN 에 의해 도출된 eyes 핵심 속성 값의 리스트
# - mouth_scores          (list)        : Property Score CNN 에 의해 도출된 mouth 핵심 속성 값의 리스트
# - pose_scores           (list)        : Property Score CNN 에 의해 도출된 pose 핵심 속성 값의 리스트
# - code_w                (Tensor)      : latent code (w) 에 해당하는 부분 (dim: 512)
# - save_dir              (str)         : 이미지를 저장할 디렉토리 경로 (stylegan_vectorfind_v8/inference_test_after_training)
# - img_file_name         (str)         : 저장할 이미지 파일 이름
# - vi                    (int)         : n vector index
# - pms                   (dict)        : eyes, mouth, pose 핵심 속성 값 변화 벡터를 latent code 에 더하거나 빼기 위한 가중치
#                                         {'eyes': float, 'mouth': float, 'pose': float}

def generate_image(finetune_v8_generator, property_score_cnn, eyes_vector, mouth_vector, pose_vector,
                   eyes_scores, mouth_scores, pose_scores, code_w, save_dir, img_file_name, vi, pms):

    eyes_pm, mouth_pm, pose_pm = pms['eyes'], pms['mouth'], pms['pose']

    # generate image
    with torch.no_grad():
        code_w_ = code_w + eyes_pm * torch.tensor(eyes_vector[vi:vi + 1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_ + mouth_pm * torch.tensor(mouth_vector[vi:vi + 1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_ + pose_pm * torch.tensor(pose_vector[vi:vi + 1, :ORIGINAL_HIDDEN_DIMS_W])
        code_w_ = code_w_.type(torch.float32)

        images = generate_image_using_w(finetune_v8_generator, code_w_)

    save_image(os.path.join(save_dir, img_file_name), images[0])

    # compute (predict) property score for each generated image using CNN
    with torch.no_grad():
        image = read_image(f'{save_dir}/{img_file_name}')
        image = stylegan_transform(image)

        property_scores = property_score_cnn(image.unsqueeze(0).cuda())
        property_scores_np = property_scores.detach().cpu().numpy()

        eyes_scores.append(round(property_scores_np[0][0], 4))
        mouth_scores.append(round(property_scores_np[0][3], 4))
        pose_scores.append(round(property_scores_np[0][4], 4))


# 이미지 50장 생성 후 비교 테스트를 위한, property score label (intermediate w vector 에 n vector 를 가감할 때의 가중치) 생성 및 반환
# Create Date : 2025.06.24
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


if __name__ == '__main__':
    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v8 : {device}')

    finetune_v8_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)  # v1, v8 Generator 는 동일한 구조

    # try loading StyleGAN-VectorFind-v8 pre-trained model
    generator_state_dict = load_existing_stylegan_vectorfind_v8(device)
    finetune_v8_generator.load_state_dict(generator_state_dict)
    print('Existing StyleGAN-VectorFind-v8 Generator load successful!! 😊')

    # get property score changing vector
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors()
    print('Existing "Property Score Changing Vector" info load successful!! 😊')

    # get Merged Property Score CNN
    property_score_cnn = load_merged_property_score_cnn(device)

    # image generation test
    finetune_v8_generator.to(device)

    run_property_score_compare_test(finetune_v8_generator,
                                    property_score_cnn,
                                    eyes_vectors,
                                    mouth_vectors,
                                    pose_vectors)
