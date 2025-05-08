from torchvision.io import read_image

from stylegan_vectorfind_v6.main import main as stylegan_vectorfind_v6_main
from stylegan_vectorfind_v6.run_vector_find import compute_medians
from stylegan_common.visualizer import postprocess_image, save_image
import stylegan_common.stylegan_generator as gen

from common import load_existing_stylegan_finetune_v1, stylegan_transform
from property_score_cnn import load_cnn_model as load_property_cnn_model

import torch
import os
import numpy as np
import pandas as pd

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256

ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS_Z = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값

TEST_IMG_CASES = 20
TEST_IMG_CASES_FOR_COMPARE_MAX = 20
TEST_IMG_CASES_NEEDED_PASS = 20

IMAGE_GENERATION_REPORT_PATH = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/image_generation_report'
os.makedirs(IMAGE_GENERATION_REPORT_PATH, exist_ok=True)

GROUP_NAMES = ['hhh', 'hhl', 'hlh', 'hll', 'lhh', 'lhl', 'llh', 'lll']
PROPERTY_NAMES = ['eyes', 'mouth', 'pose']

medians = compute_medians()  # returned values : -0.2709, 0.3052, 0.0742
kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)


# Property Score 값을 변경하기 위해 latent vector z 에 가감할 벡터 정보 반환 ('hhh', 'hhl', ..., 'lll' 의 각 그룹 별)
# Create Date : 2025.05.06
# Last Update Date : 2025.05.08
# - 생성된 이미지를 머리 색, 머리 길이, 배경 색 평균에 따라 그룹화한 것을 반영

# Arguments:
# - 없음

# Returns:
# - eyes_vectors  (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors  (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def get_property_change_vectors():
    vector_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/property_score_vectors'

    eyes_vectors = {}
    mouth_vectors = {}
    pose_vectors = {}

    for group_name in GROUP_NAMES:
        eyes_vector = np.array(pd.read_csv(f'{vector_save_dir}/eyes_change_z_vector_{group_name}.csv',
                                           index_col=0))

        mouth_vector = np.array(pd.read_csv(f'{vector_save_dir}/mouth_change_z_vector_{group_name}.csv',
                                            index_col=0))

        pose_vector = np.array(pd.read_csv(f'{vector_save_dir}/pose_change_z_vector_{group_name}.csv',
                                           index_col=0))

        eyes_vectors[group_name] = eyes_vector
        mouth_vectors[group_name] = mouth_vector
        pose_vectors[group_name] = pose_vector

    return eyes_vectors, mouth_vectors, pose_vectors


# latent code (z) 로 생성된 이미지의 group 이름 (머리 색, 머리 길이, 배경색 평균 속성값에 근거한 'hhh', 'hhl', ..., 'lll') 반환
# Create Date : 2025.05.08
# Last Update Date : -

# Arguments:
# - code_part1 (Tensor) : latent code (z) 에 해당하는 부분 (dim: 512)
# - code_part2 (Tensor) : latent code 중 원래 StyleGAN-FineTune-v1 의 핵심 속성 값 목적으로 사용된 부분 (dim: 3)
# - save_dir   (str)    : 이미지를 저장할 디렉토리 경로 (stylegan_vectorfind_v6/inference_test_after_training)
# - i          (int)    : case index
# - vi         (int)    : n vector index

# Returns:
# - group_name (str) : 이미지의 group 이름 ('hhh', 'hhl', ..., 'lll' 중 하나)

def get_group_name(code_part1, code_part2, save_dir, i, vi):
    images = finetune_v1_generator(code_part1.cuda(), code_part2.cuda(), **kwargs_val)['image']
    images = postprocess_image(images.detach().cpu().numpy())
    save_image(os.path.join(save_dir, f'original_case_{i:02d}_{vi:02d}.jpg'), images[0])

    # input generated image to Property Score CNN -> get appropriate group of generated image
    with torch.no_grad():
        image = read_image(f'{save_dir}/original_case_{i:02d}_{vi:02d}.jpg')
        image = stylegan_transform(image)

        property_scores = property_score_cnn(image.unsqueeze(0).cuda())
        property_scores_np = property_scores.detach().cpu().numpy()

    hair_color_group = 'h' if property_scores_np[0][1] >= medians['hair_color'] else 'l'
    hair_length_group = 'h' if property_scores_np[0][2] >= medians['hair_length'] else 'l'
    background_mean_group = 'h' if property_scores_np[0][5] >= medians['background_mean'] else 'l'

    group_name = hair_color_group + hair_length_group + background_mean_group
    return group_name


# latent vector z 에 가감할 Property Score Vector 를 이용한 Property Score 값 변화 테스트 (이미지 생성 테스트)
# Create Date : 2025.05.06
# Last Update Date : 2025.05.08
# - 생성된 이미지를 머리 색, 머리 길이, 배경 색 평균에 따라 그룹화한 것을 반영

# Arguments:
# - finetune_v1_generator (nn.Module)         : StyleGAN-FineTune-v1 의 Generator
# - property_score_cnn    (nn.Module)         : 핵심 속성 값을 계산하기 위한 CNN
# - eyes_vectors          (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors         (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors          (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

# Returns:
# - stylegan_vectorfind_v6/inference_test_after_training 디렉토리에 이미지 생성 결과 저장

def run_image_generation_test(finetune_v1_generator, property_score_cnn, eyes_vectors, mouth_vectors, pose_vectors):
    save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/inference_test_after_training'
    os.makedirs(save_dir, exist_ok=True)

    n_vector_cnt = len(eyes_vectors['hhh'])  # equal to pre-defined SVMS_PER_EACH_PROPERTY value
    vector_dicts = [eyes_vectors, mouth_vectors, pose_vectors]

    for i in range(TEST_IMG_CASES):
        code_part1 = torch.randn(1, ORIGINAL_HIDDEN_DIMS_Z)      # 512
        code_part2 = torch.randn(1, ORIGINALLY_PROPERTY_DIMS_Z)  # 3

        for vi in range(n_vector_cnt):
            group_name = get_group_name(code_part1, code_part2, save_dir, i, vi)

            # run image generation test
            for property_name, vector_dict in zip(PROPERTY_NAMES, vector_dicts):
                vector = vector_dict[group_name]
                pms = [-2.0, -0.67, 0.67, 2.0]

                for pm_idx, pm in enumerate(pms):
                    with torch.no_grad():
                        code_part1_ = code_part1 + pm * torch.tensor(vector[vi:vi+1, :ORIGINAL_HIDDEN_DIMS_Z])  # 512
                        code_part2_ = code_part2 + pm * torch.tensor(vector[vi:vi+1, ORIGINAL_HIDDEN_DIMS_Z:])  # 3
                        code_part1_ = code_part1_.type(torch.float32)
                        code_part2_ = code_part2_.type(torch.float32)

                        images = finetune_v1_generator(code_part1_.cuda(), code_part2_.cuda(), **kwargs_val)['image']
                        images = postprocess_image(images.detach().cpu().numpy())

                        save_image(os.path.join(save_dir, f'case_{i:02d}_{vi:02d}_{property_name}_pm_{pm_idx}.jpg'),
                                   images[0])


# 이미지 50장 생성 후 의도한 property score label 과, 생성된 이미지에 대한 CNN 예측 property score 를 비교 테스트 (corr-coef)
# Create Date : 2025.05.07
# Last Update Date : 2025.05.08
# - 정해진 PASSED (비교 테스트 합격) 케이스 개수를 채울 때까지 반복하는 메커니즘 적용
# - 이미지 생성 도중 각 케이스에 대한 테스트 결과 출력
# - 생성된 이미지를 머리 색, 머리 길이, 배경 색 평균에 따라 그룹화한 것을 반영

# Arguments:
# - finetune_v1_generator (nn.Module)         : StyleGAN-FineTune-v1 의 Generator
# - property_score_cnn    (nn.Module)         : 핵심 속성 값을 계산하기 위한 CNN
# - eyes_vectors          (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors         (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors          (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

# Returns:
# - stylegan_vectorfind_v6/inference_test_after_training 디렉토리에 이미지 생성
# - stylegan_vectorfind_v6/image_generation_report 디렉토리에 테스트 결과를 csv 파일로 저장

def run_property_score_compare_test(finetune_v1_generator, property_score_cnn, eyes_vectors, mouth_vectors,
                                    pose_vectors):

    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    n_vector_cnt = len(eyes_vectors['hhh'])  # equal to pre-defined SVMS_PER_EACH_PROPERTY value
    passed_count = 0

    # label: 'eyes', 'mouth', 'pose'
    eyes_pm_order, mouth_pm_order, pose_pm_order = get_pm_labels()
    pm_cnt = len(eyes_pm_order)

    all_data_dict = {'case': [], 'vector_no': [], 'passed': [],
                     'eyes_corr': [], 'mouth_corr': [], 'pose_corr': []}

    code_part1s_np = np.zeros((TEST_IMG_CASES_FOR_COMPARE_MAX, ORIGINAL_HIDDEN_DIMS_Z))
    code_part2s_np = np.zeros((TEST_IMG_CASES_FOR_COMPARE_MAX, ORIGINALLY_PROPERTY_DIMS_Z))
    generated_count = 0

    # image generation
    for i in range(TEST_IMG_CASES_FOR_COMPARE_MAX):
        save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/inference_test_after_training/test_{i:04d}'
        os.makedirs(save_dir, exist_ok=True)

        code_part1 = torch.randn(1, ORIGINAL_HIDDEN_DIMS_Z)      # 512
        code_part2 = torch.randn(1, ORIGINALLY_PROPERTY_DIMS_Z)  # 3
        code_part1s_np[i] = code_part1[0]
        code_part2s_np[i] = code_part2[0]

        for vi in range(n_vector_cnt):
            group_name = get_group_name(code_part1, code_part2, save_dir, i, vi)

            eyes_vector = eyes_vectors[group_name]
            mouth_vector = mouth_vectors[group_name]
            pose_vector = pose_vectors[group_name]

            eyes_scores, mouth_scores, pose_scores = [], [], []

            all_data_dict['case'].append(i)
            all_data_dict['vector_no'].append(vi)

            for pm_idx in range(pm_cnt):
                img_file_name = f'case_{i:03d}_{vi:03d}_pm_{pm_idx:03d}.jpg'

                eyes_pm = eyes_pm_order[pm_idx]
                mouth_pm = mouth_pm_order[pm_idx]
                pose_pm = pose_pm_order[pm_idx]

                # generate image
                with torch.no_grad():
                    code_part1_ = code_part1 + eyes_pm * torch.tensor(eyes_vector[vi:vi+1, :ORIGINAL_HIDDEN_DIMS_Z])
                    code_part1_ = code_part1_ + mouth_pm * torch.tensor(mouth_vector[vi:vi+1, :ORIGINAL_HIDDEN_DIMS_Z])
                    code_part1_ = code_part1_ + pose_pm * torch.tensor(pose_vector[vi:vi+1, :ORIGINAL_HIDDEN_DIMS_Z])
                    code_part1_ = code_part1_.type(torch.float32)

                    code_part2_ = code_part2 + eyes_pm * torch.tensor(eyes_vector[vi:vi+1, ORIGINAL_HIDDEN_DIMS_Z:])
                    code_part2_ = code_part2_ + mouth_pm * torch.tensor(mouth_vector[vi:vi+1, ORIGINAL_HIDDEN_DIMS_Z:])
                    code_part2_ = code_part2_ + pose_pm * torch.tensor(pose_vector[vi:vi+1, ORIGINAL_HIDDEN_DIMS_Z:])
                    code_part2_ = code_part2_.type(torch.float32)

                    images = finetune_v1_generator(code_part1_.cuda(), code_part2_.cuda(), **kwargs_val)['image']
                    images = postprocess_image(images.detach().cpu().numpy())

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

            # compute and record corr-coef
            eyes_corrcoef = np.corrcoef(eyes_pm_order, eyes_scores)[0][1]
            mouth_corrcoef = np.corrcoef(mouth_pm_order, mouth_scores)[0][1]
            pose_corrcoef = np.corrcoef(pose_pm_order, pose_scores)[0][1]

            all_data_dict['eyes_corr'].append(round(eyes_corrcoef, 4))
            all_data_dict['mouth_corr'].append(round(mouth_corrcoef, 4))
            all_data_dict['pose_corr'].append(round(pose_corrcoef, 4))

            # check passed
            generated_count += 1

            passed = abs(eyes_corrcoef) >= 0.75 and abs(mouth_corrcoef) >= 0.77 and abs(pose_corrcoef) >= 0.8
            eyes_diff = max(0.75 - abs(eyes_corrcoef), 0)
            mouth_diff = max(0.77 - abs(mouth_corrcoef), 0)
            pose_diff = max(0.8 - abs(pose_corrcoef), 0)

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

            print(f'testing idx {i} vector {vi} ... (passed : {passed_count}, current margin: {round(pass_diff, 4)}, '
                  f'diff: {diff})')

        if passed_count >= TEST_IMG_CASES_NEEDED_PASS:
            break

    # save all data
    all_data_df = pd.DataFrame(all_data_dict)
    all_data_df['sum_abs_corr'] = abs(all_data_df['eyes_corr']) + abs(all_data_df['mouth_corr']) + abs(all_data_df['pose_corr'])
    all_data_df['sum_abs_corr'] = all_data_df['sum_abs_corr'].apply(lambda x: round(x, 4))

    all_data_save_path = f'{IMAGE_GENERATION_REPORT_PATH}/test_result.csv'
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

    statistics_save_path = f'{IMAGE_GENERATION_REPORT_PATH}/test_statistics.csv'
    statistics_df.to_csv(statistics_save_path)

    # save latent codes (z)
    code_part1s_np = np.round(code_part1s_np[:generated_count], 4)
    code_part2s_np = np.round(code_part2s_np[:generated_count], 4)
    code_all_np = np.concatenate([code_part1s_np, code_part2s_np], axis=1)

    pd.DataFrame(code_part1s_np).to_csv(f'{IMAGE_GENERATION_REPORT_PATH}/latent_codes_part1.csv', index=False)
    pd.DataFrame(code_part2s_np).to_csv(f'{IMAGE_GENERATION_REPORT_PATH}/latent_codes_part2.csv', index=False)
    pd.DataFrame(code_all_np).to_csv(f'{IMAGE_GENERATION_REPORT_PATH}/latent_codes_all.csv', index=False)


# 이미지 50장 생성 후 비교 테스트를 위한, property score label (latent z vector 에 n vector 를 가감할 때의 가중치) 생성 및 반환
# Create Date : 2025.05.07
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - eyes_pm_order  (list(float)) : eyes (눈을 뜬 정도) 속성에 대한 50장 각각의 property score label
# - mouth_pm_order (list(float)) : mouth (입을 벌린 정도) 속성에 대한 50장 각각의 property score label
# - pose_pm_order  (list(float)) : pose (고개 돌림) 속성에 대한 50장 각각의 property score label

def get_pm_labels():
    eyes_pms = [-1.8, 2.4]
    mouth_pms = [-2.8, -1.4, 0.0, 1.4, 2.8]
    pose_pms = [-2.1, -1.4, -0.7, 0.0, 0.7]

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v1 : {device}')

    finetune_v1_generator = gen.StyleGANGeneratorForV6(resolution=IMAGE_RESOLUTION)
    generator_state_dict = load_existing_stylegan_finetune_v1(device)
    print('Existing StyleGAN-FineTune-v1 Generator load successful!! 😊')

    # load state dict (generator)
    del generator_state_dict['mapping.label_weight']  # size mismatch because of modified property vector dim (7 -> 3)
    finetune_v1_generator.load_state_dict(generator_state_dict, strict=False)

    # get property score changing vector
    try:
        eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors()

    except:
        stylegan_vectorfind_v6_main(finetune_v1_generator, device)
        eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors()

    # get Property Score CNN
    property_cnn_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_score_cnn = load_property_cnn_model(property_cnn_path, device)

    # image generation test
    finetune_v1_generator.to(device)

    run_image_generation_test(finetune_v1_generator,
                              property_score_cnn,
                              eyes_vectors,
                              mouth_vectors,
                              pose_vectors)

    run_property_score_compare_test(finetune_v1_generator,
                                    property_score_cnn,
                                    eyes_vectors,
                                    mouth_vectors,
                                    pose_vectors)
