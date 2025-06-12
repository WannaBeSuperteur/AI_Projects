try:
    from stylegan_vectorfind_v9.main import main_svm as stylegan_vectorfind_v9_main_svm
    from stylegan_vectorfind_v9.main import main_gradient as stylegan_vectorfind_v9_main_gradient
    from run_stylegan_vectorfind_v9_svm import load_ohlora_mid_vector_group_names, get_group_name
    from stylegan_common.visualizer import save_image
    import stylegan_common.stylegan_generator as gen

    from common import (load_existing_stylegan_finetune_v9,
                        load_existing_stylegan_vectorfind_v9,
                        load_merged_property_score_cnn,
                        stylegan_transform)
    from common_vectorfind_v9 import (generate_image_using_mid_vector,
                                      generate_code_mid,
                                      load_ohlora_z_vectors,
                                      get_pm_labels)

except:
    from stylegan.stylegan_vectorfind_v9.main import main_svm as stylegan_vectorfind_v9_main_svm
    from stylegan.stylegan_vectorfind_v9.main import main_gradient as stylegan_vectorfind_v9_main_gradient
    from stylegan.run_stylegan_vectorfind_v9_svm import load_ohlora_mid_vector_group_names, get_group_name
    from stylegan.stylegan_common.visualizer import save_image
    import stylegan.stylegan_common.stylegan_generator as gen

    from stylegan.common import (load_existing_stylegan_finetune_v9,
                                 load_existing_stylegan_vectorfind_v9,
                                 load_merged_property_score_cnn,
                                 stylegan_transform)
    from stylegan.common_vectorfind_v9 import (generate_image_using_mid_vector,
                                               generate_code_mid,
                                               load_ohlora_z_vectors,
                                               get_pm_labels)

from run_stylegan_vectorfind_v9_svm import get_property_change_vectors
from run_stylegan_vectorfind_v9_svm import run_image_generation_test as run_image_generation_test_svm
from run_stylegan_vectorfind_v9_svm import run_property_score_compare_test as run_property_score_compare_test_svm

from run_stylegan_vectorfind_v9_gradient import get_property_change_gradient_nn
from run_stylegan_vectorfind_v9_gradient import run_image_generation_test as run_image_generation_test_gradient
from run_stylegan_vectorfind_v9_gradient import run_property_score_compare_test as run_property_score_compare_test_gradient

import torch
from torchvision.io import read_image
import numpy as np
import pandas as pd

import os
import shutil
import time


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256
finetune_v9_generator = None
PROPERTY_NAMES = ['eyes', 'mouth', 'pose']

ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINAL_HIDDEN_DIMS_W = 512
HIDDEN_DIMS_MAPPING_SPLIT1 = 512 + 2048
HIDDEN_DIMS_MAPPING_SPLIT2 = 512 + 512

ORIGINALLY_PROPERTY_DIMS = 7    # 원래 property (eyes, hair_color, hair_length, mouth, pose,
                                #               background_mean, background_std) 목적으로 사용된 dimension 값

TEST_IMG_CASES = 1
TEST_IMG_CASES_FOR_COMPARE_MAX = 100  # 2000
TEST_IMG_CASES_NEEDED_PASS = 100  # 60


test_result_svm = {'n': [], 'k': [], 'time': [],
                   'svm_eyes_acc': [], 'svm_mouth_acc': [], 'svm_pose_acc': [],
                   'eyes_mean_corr': [], 'mouth_mean_corr': [], 'pose_mean_corr': [], 'sum_mean_corr': []}

test_result_grad = {'n': [], 'time': [],
                    'nn_eyes_mse': [], 'nn_mouth_mse': [], 'nn_pose_mse': [],
                    'eyes_mean_corr': [], 'mouth_mean_corr': [], 'pose_mean_corr': [], 'sum_mean_corr': []}

test_result_final = {'n': [], 'k': [], 'time': [],
                     'nn_eyes_mse': [], 'svm_mouth_acc': [], 'svm_pose_acc': [],
                     'eyes_mean_corr': [], 'mouth_mean_corr': [], 'pose_mean_corr': [], 'sum_mean_corr': []}


IMAGE_GENERATION_REPORT_PATH = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/image_generation_report'
OHLORA_FINAL_VECTORS_TEST_REPORT_PATH = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/final_vector_test_report'
os.makedirs(IMAGE_GENERATION_REPORT_PATH, exist_ok=True)
os.makedirs(OHLORA_FINAL_VECTORS_TEST_REPORT_PATH, exist_ok=True)

image_gen_report_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/image_generation_report'
vector_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/property_score_vectors'
generated_img_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/inference_test_after_training'
test_result_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/test_result'
models_dir = f'{PROJECT_DIR_PATH}/stylegan/models'
fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/models'

os.makedirs(test_result_dir, exist_ok=True)


# StyleGAN-VectorFind-v9 자동화 테스트 함수 (SVM 기반)
# Create Date : 2025.06.10
# Last Update Date : 2025.06.11
# - import 된 함수명 수정 반영

# Arguments:
# - n          (int)   : 총 생성할 이미지 sample 개수
# - ratio      (float) : 총 생성 이미지 중 SVM 의 학습 데이터로 사용할 TOP, BOTTOM 비율 (각각) (= k / n)
# - layer_name (str)   : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름
#                        ('mapping_split1', 'mapping_split2' or 'w')

def run_stylegan_vectorfind_v9_automated_test_svm(n, ratio, layer_name):
    global finetune_v9_generator

    start_at = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v9 : {device}')

    # load StyleGAN-FineTune-v9 pre-trained model
    finetune_v9_generator = gen.StyleGANGeneratorForV9(resolution=IMAGE_RESOLUTION)  # v1, v9 Generator 는 동일한 구조
    generator_state_dict = load_existing_stylegan_finetune_v9(device)
    print('Existing StyleGAN-FineTune-v9 Generator load successful!! 😊')

    finetune_v9_generator.load_state_dict(generator_state_dict)

    # get property score changing vector
    entire_svm_accuracy_dict = stylegan_vectorfind_v9_main_svm(finetune_v9_generator, device, n, ratio, layer_name)
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors(layer_name)

    # get Merged Property Score CNN
    property_score_cnn = load_merged_property_score_cnn(device)

    # image generation test
    finetune_v9_generator.to(device)

    run_image_generation_test_svm(finetune_v9_generator,
                                  property_score_cnn,
                                  layer_name,
                                  eyes_vectors,
                                  mouth_vectors,
                                  pose_vectors)

    eyes_corr_mean, mouth_corr_mean, pose_corr_mean = run_property_score_compare_test_svm(finetune_v9_generator,
                                                                                          property_score_cnn,
                                                                                          layer_name,
                                                                                          eyes_vectors,
                                                                                          mouth_vectors,
                                                                                          pose_vectors)

    # add experiment log
    elapsed_time = time.time() - start_at

    test_result_svm['n'].append(n)
    test_result_svm['k'].append(int(round(n * ratio)))
    test_result_svm['time'].append(round(elapsed_time, 2))

    test_result_svm['svm_eyes_acc'].append(round(entire_svm_accuracy_dict['eyes'], 4))
    test_result_svm['svm_mouth_acc'].append(round(entire_svm_accuracy_dict['mouth'], 4))
    test_result_svm['svm_pose_acc'].append(round(entire_svm_accuracy_dict['pose'], 4))

    sum_mean_corr = abs(round(eyes_corr_mean, 4)) + abs(round(mouth_corr_mean, 4)) + abs(round(pose_corr_mean, 4))
    test_result_svm['eyes_mean_corr'].append(abs(round(eyes_corr_mean, 4)))
    test_result_svm['mouth_mean_corr'].append(abs(round(mouth_corr_mean, 4)))
    test_result_svm['pose_mean_corr'].append(abs(round(pose_corr_mean, 4)))
    test_result_svm['sum_mean_corr'].append(sum_mean_corr)

    test_result_svm_df = pd.DataFrame(test_result_svm)
    test_result_svm_df.to_csv(f'{test_result_dir}/test_result_svm_{layer_name}.csv')

    # re-initialize test directories
    shutil.rmtree(image_gen_report_dir)
    shutil.rmtree(vector_save_dir)
    shutil.rmtree(generated_img_dir)

    os.makedirs(image_gen_report_dir)
    os.makedirs(vector_save_dir)
    os.makedirs(generated_img_dir)


# StyleGAN-VectorFind-v9 자동화 테스트 함수 (Gradient Neural Network 기반)
# Create Date : 2025.06.11
# Last Update Date : 2025.06.11
# - Gradient NN 가중치 파일 삭제 오류 해결
# - image generation report 디렉토리 삭제 -> 유지로 변경
# - Gradient NN 방법론 특성상 vector 저장 디렉토리가 없음에도 불구하고 삭제 시도하는 버그 해결

# Arguments:
# - n          (int) : 총 생성할 이미지 sample 개수
# - layer_name (str) : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름 ('mapping_split1', 'mapping_split2' or 'w')

def run_stylegan_vectorfind_v9_automated_test_gradient(n, layer_name):
    global finetune_v9_generator

    start_at = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v9 : {device}')

    # load StyleGAN-FineTune-v9 pre-trained model
    finetune_v9_generator = gen.StyleGANGeneratorForV9(resolution=IMAGE_RESOLUTION)  # v1, v9 Generator 는 동일한 구조
    generator_state_dict = load_existing_stylegan_finetune_v9(device)
    print('Existing StyleGAN-FineTune-v9 Generator load successful!! 😊')

    finetune_v9_generator.load_state_dict(generator_state_dict)

    # save gradient neural network
    mse_errors = stylegan_vectorfind_v9_main_gradient(finetune_v9_generator, device, n, layer_name)
    eyes_gradient_nn = get_property_change_gradient_nn(property_name='eyes', layer_name=layer_name)
    mouth_gradient_nn = get_property_change_gradient_nn(property_name='mouth', layer_name=layer_name)
    pose_gradient_nn = get_property_change_gradient_nn(property_name='pose', layer_name=layer_name)

    # get Merged Property Score CNN
    property_score_cnn = load_merged_property_score_cnn(device)

    # image generation test
    finetune_v9_generator.to(device)

    run_image_generation_test_gradient(finetune_v9_generator,
                                       layer_name=layer_name,
                                       eyes_gradient_nn=eyes_gradient_nn,
                                       mouth_gradient_nn=mouth_gradient_nn,
                                       pose_gradient_nn=pose_gradient_nn)

    eyes_corr_mean, mouth_corr_mean, pose_corr_mean = (
        run_property_score_compare_test_gradient(finetune_v9_generator,
                                                 property_score_cnn,
                                                 n=n,
                                                 layer_name=layer_name,
                                                 eyes_gradient_nn=eyes_gradient_nn,
                                                 mouth_gradient_nn=mouth_gradient_nn,
                                                 pose_gradient_nn=pose_gradient_nn))

    # add experiment log
    elapsed_time = time.time() - start_at

    test_result_grad['n'].append(n)
    test_result_grad['time'].append(round(elapsed_time, 2))

    test_result_grad['nn_eyes_mse'].append(round(mse_errors['eyes'], 4))
    test_result_grad['nn_mouth_mse'].append(round(mse_errors['mouth'], 4))
    test_result_grad['nn_pose_mse'].append(round(mse_errors['pose'], 4))

    sum_mean_corr = abs(round(eyes_corr_mean, 4)) + abs(round(mouth_corr_mean, 4)) + abs(round(pose_corr_mean, 4))
    test_result_grad['eyes_mean_corr'].append(abs(round(eyes_corr_mean, 4)))
    test_result_grad['mouth_mean_corr'].append(abs(round(mouth_corr_mean, 4)))
    test_result_grad['pose_mean_corr'].append(abs(round(pose_corr_mean, 4)))
    test_result_grad['sum_mean_corr'].append(sum_mean_corr)

    test_result_grad_df = pd.DataFrame(test_result_grad)
    test_result_grad_df.to_csv(f'{test_result_dir}/test_result_gradient_{layer_name}.csv')

    # re-initialize test directories
    shutil.rmtree(generated_img_dir)
    os.makedirs(generated_img_dir)

    # remove NN models for gradients
    for property_name in PROPERTY_NAMES:
        model_path = f'{models_dir}/stylegan_gen_vector_find_v9_nn_{property_name}.pth'
        if os.path.isfile(model_path):
            os.remove(model_path)


# 주어진 eyes, mouth, pose 핵심 속성 값 변화 벡터를 이용하여 이미지 생성
# Create Date : 2025.06.12
# Last Update Date : -

# Arguments:
# - finetune_v9_generator (nn.Module)   : StyleGAN-FineTune-v9 의 Generator
# - property_score_cnn    (nn.Module)   : 핵심 속성 값을 계산하기 위한 CNN
# - eyes_vector           (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터
# - mouth_vector          (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터
# - pose_vector           (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터
# - eyes_scores           (list)        : Property Score CNN 에 의해 도출된 eyes 핵심 속성 값의 리스트
# - mouth_scores          (list)        : Property Score CNN 에 의해 도출된 mouth 핵심 속성 값의 리스트
# - pose_scores           (list)        : Property Score CNN 에 의해 도출된 pose 핵심 속성 값의 리스트
# - code_mid              (Tensor)      : latent code (intermediate vector) 에 해당하는 부분
# - layer_name            (str)         : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름
#                                         ('mapping_split1', 'mapping_split2' or 'w')
# - save_dir              (str)         : 이미지를 저장할 디렉토리 경로 (stylegan_vectorfind_v9/inference_test_after_training)
# - img_file_name         (str)         : 저장할 이미지 파일 이름
# - pms                   (dict)        : eyes, mouth, pose 핵심 속성 값 변화 벡터를 latent code 에 더하거나 빼기 위한 가중치
#                                         {'eyes': float, 'mouth': float, 'pose': float}

def generate_image_final(finetune_v9_generator, property_score_cnn, eyes_vector, mouth_vector, pose_vector,
                         eyes_scores, mouth_scores, pose_scores, code_mid, layer_name, save_dir, img_file_name, pms):

    eyes_pm, mouth_pm, pose_pm = pms['eyes'], pms['mouth'], pms['pose']

    # generate image
    if layer_name == 'w':
        dim = ORIGINAL_HIDDEN_DIMS_W
    elif layer_name == 'mapping_split1':
        dim = HIDDEN_DIMS_MAPPING_SPLIT1
    else:  # mapping_split2
        dim = HIDDEN_DIMS_MAPPING_SPLIT2

    with torch.no_grad():
        code_mid_ = code_mid + eyes_pm * torch.tensor(eyes_vector[0:1, :dim])
        code_mid_ = code_mid_ + mouth_pm * torch.tensor(mouth_vector[0:1, :dim])
        code_mid_ = code_mid_ + pose_pm * torch.tensor(pose_vector[0:1, :dim])
        code_mid_ = code_mid_.type(torch.float32)

        images = generate_image_using_mid_vector(finetune_v9_generator, code_mid_, layer_name)

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


# 이미지 50장 생성 후 의도한 property score label 과, 생성된 이미지에 대한 CNN 예측 property score 를 비교 테스트 (corr-coef)
# Create Date : 2025.06.12
# Last Update Date : -

# Arguments:
# - finetune_v9_generator (nn.Module) : StyleGAN-FineTune-v9 의 Generator
# - property_score_cnn    (nn.Module) : 핵심 속성 값을 계산하기 위한 CNN
# - n                     (int)       : 총 생성할 이미지 샘플 개수
# - layer_name            (str)       : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름
#                                       ('mapping_split1', 'mapping_split2' or 'w')
# - eyes_gradient_nn      (nn.Module) : eyes (눈을 뜬 정도) 의 Gradient (= 핵심 속성 값 변화 벡터) 탐색 용 딥러닝 모델
# - mouth_gradient_nn     (nn.Module) : mouth (입을 벌린 정도) 의 Gradient (= 핵심 속성 값 변화 벡터) 탐색 용 딥러닝 모델
# - pose_gradient_nn      (nn.Module) : pose (고개 돌림) 의 Gradient (= 핵심 속성 값 변화 벡터) 탐색 용 딥러닝 모델

# Returns:
# - eyes_corr_mean (float)             : eyes (눈을 뜬 정도) 속성값에 대한 "의도한 값" - "생성된 이미지 실측값" 에 대한 실측값의 상관계수 평균
# - mouth_vectors  (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors   (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

# File Outputs:
# - stylegan_vectorfind_v9/inference_test_after_training 디렉토리에 이미지 생성
# - stylegan_vectorfind_v9/image_generation_report 디렉토리에 테스트 결과를 csv 파일로 저장

def run_property_score_compare_test_final(finetune_v9_generator, property_score_cnn, n, layer_name, eyes_gradient_nn,
                                          mouth_vectors, pose_vectors):

    ohlora_z_vector_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/ohlora_z_vectors.csv'
    ohlora_mid_group_name_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/ohlora_{layer_name}_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_mid_group_names = load_ohlora_mid_vector_group_names(group_name_csv_path=ohlora_mid_group_name_csv_path,
                                                                layer_name=layer_name)

    n_kilo = f'{n // 1000}K'
    passed_count = 0

    # label: 'eyes', 'mouth', 'pose'
    eyes_pms = [-1.2, 1.2]
    mouth_pms = [-1.6, -0.8, 0.0, 0.8, 1.6]
    pose_pms = [-1.4, -0.7, 0.0, 0.7, 1.4]

    eyes_pm_order, mouth_pm_order, pose_pm_order = get_pm_labels(eyes_pms, mouth_pms, pose_pms)
    pm_cnt = len(eyes_pm_order)

    all_data_dict = {'case': [], 'svm_group_name': [], 'passed': [],
                     'eyes_corr': [], 'mouth_corr': [], 'pose_corr': []}

    if ohlora_z_vectors is not None:
        count_to_generate = len(ohlora_z_vectors)
    else:
        count_to_generate = TEST_IMG_CASES_FOR_COMPARE_MAX

    code_part1s_np = np.zeros((count_to_generate, ORIGINAL_HIDDEN_DIMS_Z))
    code_part2s_np = np.zeros((count_to_generate, ORIGINALLY_PROPERTY_DIMS))
    generated_count = 0

    # image generation
    for i in range(count_to_generate):
        save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/inference_test_after_training/test_{i:04d}'
        os.makedirs(save_dir, exist_ok=True)

        if ohlora_z_vectors is not None:
            code_part1s_np[i] = ohlora_z_vectors[i][:ORIGINAL_HIDDEN_DIMS_Z]
            code_part2s_np[i] = ohlora_z_vectors[i][ORIGINAL_HIDDEN_DIMS_Z:]
            code_part1 = torch.tensor(code_part1s_np[i]).unsqueeze(0).to(torch.float32)  # 512
            code_part2 = torch.tensor(code_part2s_np[i]).unsqueeze(0).to(torch.float32)  # 7

        else:
            code_part1 = torch.randn(1, ORIGINAL_HIDDEN_DIMS_Z)    # 512
            code_part2 = torch.randn(1, ORIGINALLY_PROPERTY_DIMS)  # 7
            code_part1s_np[i] = code_part1[0]
            code_part2s_np[i] = code_part2[0]

        code_mid = generate_code_mid(finetune_v9_generator, layer_name, code_part1, code_part2)
        eyes_scores, mouth_scores, pose_scores = [], [], []
        all_data_dict['case'].append(i)

        property_change_vector_dict = {}

        # compute eyes change vector (using Gradient method)
        code_mid.requires_grad_()
        output = eyes_gradient_nn(code_mid.cuda())
        output.backward()
        gradient = code_mid.grad.detach().cpu()
        gradient = gradient / np.linalg.norm(gradient)

        output = output.detach().cpu()      # to prevent memory leak
        code_mid = code_mid.detach().cpu()  # to prevent memory leak

        property_change_vector_dict['eyes'] = gradient

        # compute mouth, pose change vector (using SVM method)
        if ohlora_mid_group_names is None:
            group_name = get_group_name(finetune_v9_generator, property_score_cnn,
                                        code_part1, code_part2, save_dir, i, vi=0)
        else:
            group_name = ohlora_mid_group_names[i]

        mouth_vector = mouth_vectors[group_name]
        pose_vector = pose_vectors[group_name]
        all_data_dict['svm_group_name'].append(group_name)

        # generate image
        for pm_idx in range(pm_cnt):
            img_file_name = f'case_{i:04d}_pm_{pm_idx:04d}.jpg'
            pms = {'eyes': eyes_pm_order[pm_idx], 'mouth': mouth_pm_order[pm_idx], 'pose': pose_pm_order[pm_idx]}

            eyes_vector = property_change_vector_dict['eyes']
            generate_image_final(finetune_v9_generator, property_score_cnn, eyes_vector, mouth_vector, pose_vector,
                                 eyes_scores, mouth_scores, pose_scores, code_mid, layer_name, save_dir, img_file_name,
                                 pms)

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

        case_data_save_path = f'{save_dir}/case_{i:04d}_result.csv'
        case_data_df.to_csv(case_data_save_path, index=False)

        print(f'testing idx {i} (passed : {passed_count}, current total gap: {round(pass_diff, 4)}, diff: {diff})')

        if ohlora_z_vectors is None and passed_count >= TEST_IMG_CASES_NEEDED_PASS:
            break

    if ohlora_z_vectors is not None:
        print('Already loaded "saved z vectors info" for Oh-LoRA face image generation.')
        image_gen_report_path = OHLORA_FINAL_VECTORS_TEST_REPORT_PATH
    else:
        image_gen_report_path = IMAGE_GENERATION_REPORT_PATH

    # save all data
    all_data_df = pd.DataFrame(all_data_dict)
    all_data_df['sum_abs_corr'] = abs(all_data_df['eyes_corr']) + abs(all_data_df['mouth_corr']) + abs(all_data_df['pose_corr'])
    all_data_df['sum_abs_corr'] = all_data_df['sum_abs_corr'].apply(lambda x: round(x, 4))

    all_data_save_path = f'{image_gen_report_path}/test_result_{layer_name}_{n_kilo}.csv'
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
                                  'passed_ratio': passed_count / generated_count})

    statistics_save_path = f'{image_gen_report_path}/test_statistics_{layer_name}_{n_kilo}.csv'
    statistics_df.to_csv(statistics_save_path)

    # save latent codes (intermediate vector)
    code_part1s_np = np.round(code_part1s_np[:generated_count], 4)
    code_part2s_np = np.round(code_part2s_np[:generated_count], 4)
    code_all_np = np.concatenate([code_part1s_np, code_part2s_np], axis=1)

    pd.DataFrame(code_part1s_np).to_csv(f'{image_gen_report_path}/latent_codes_part1_{layer_name}_{n_kilo}.csv',
                                        index=False)
    pd.DataFrame(code_part2s_np).to_csv(f'{image_gen_report_path}/latent_codes_part2_{layer_name}_{n_kilo}.csv',
                                        index=False)
    pd.DataFrame(code_all_np).to_csv(f'{image_gen_report_path}/latent_codes_all_{layer_name}_{n_kilo}.csv',
                                     index=False)

    return eyes_corr_mean, mouth_corr_mean, pose_corr_mean


# StyleGAN-VectorFind-v9 자동화 테스트 함수 (image generation report 의 final decision 기반)
# Create Date : 2025.06.12
# Last Update Date : -

# Arguments:
# - n              (int)   : 총 생성할 이미지 sample 개수
# - ratio          (float) : SVM 방법론 사용 시, 총 생성 이미지 중 학습 데이터로 사용할 TOP, BOTTOM 비율 (각각) (= k / n)
# - save_generator (bool)  : StyleGAN-VectorFind-v9 모델 저장 여부
# - remove_data    (bool)  : 실험 종료 후 실험 진행한 디렉토리 및 모델 삭제 여부

def run_stylegan_vectorfind_v9_automated_test_final(n, ratio, save_generator=False, remove_data=True):
    global finetune_v9_generator, fine_tuned_model_path
    common_layer_name = 'mapping_split1'

    start_at = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v9 : {device}')

    # load Merged Property Score CNN
    property_score_cnn = load_merged_property_score_cnn(device)

    # load StyleGAN-FineTune-v9 pre-trained model
    finetune_v9_generator = gen.StyleGANGeneratorForV9(resolution=IMAGE_RESOLUTION)  # v1, v9 Generator 는 동일한 구조
    generator_state_dict = load_existing_stylegan_finetune_v9(device)
    print('Existing StyleGAN-FineTune-v9 Generator load successful!! 😊')

    finetune_v9_generator.load_state_dict(generator_state_dict)
    finetune_v9_generator.to(device)

    # save StyleGAN-VectorFind-v9 state dict
    if save_generator:
        torch.save(finetune_v9_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_vector_find_v9.pth')

    # use Gradient method for 'eyes'
    mse_errors = stylegan_vectorfind_v9_main_gradient(finetune_v9_generator, device, n,
                                                      layer_name=common_layer_name,
                                                      property_names=['eyes'])

    eyes_gradient_nn = get_property_change_gradient_nn(property_name='eyes',
                                                       layer_name=common_layer_name)

    # use SVM method for 'mouth' and 'pose'
    entire_svm_accuracy_dict_mouth = stylegan_vectorfind_v9_main_svm(finetune_v9_generator, device, n, ratio,
                                                                     layer_name=common_layer_name)
    _, mouth_vectors, _ = get_property_change_vectors(layer_name=common_layer_name)

    entire_svm_accuracy_dict_pose = stylegan_vectorfind_v9_main_svm(finetune_v9_generator, device, n, ratio,
                                                                    layer_name=common_layer_name)
    _, _, pose_vectors = get_property_change_vectors(layer_name=common_layer_name)

    # run final image generation test
    eyes_corr_mean, mouth_corr_mean, pose_corr_mean = (
        run_property_score_compare_test_final(finetune_v9_generator,
                                              property_score_cnn,
                                              n=n,
                                              layer_name=common_layer_name,
                                              eyes_gradient_nn=eyes_gradient_nn,
                                              mouth_vectors=mouth_vectors,
                                              pose_vectors=pose_vectors))

    # add experiment log
    elapsed_time = time.time() - start_at

    test_result_grad['n'].append(n)
    test_result_grad['time'].append(round(elapsed_time, 2))

    test_result_grad['nn_eyes_mse'].append(round(mse_errors['eyes'], 4))
    test_result_grad['svm_mouth_acc'].append(round(entire_svm_accuracy_dict_mouth['mouth'], 4))
    test_result_grad['svm_pose_acc'].append(round(entire_svm_accuracy_dict_pose['pose'], 4))

    sum_mean_corr = abs(round(eyes_corr_mean, 4)) + abs(round(mouth_corr_mean, 4)) + abs(round(pose_corr_mean, 4))
    test_result_grad['eyes_mean_corr'].append(abs(round(eyes_corr_mean, 4)))
    test_result_grad['mouth_mean_corr'].append(abs(round(mouth_corr_mean, 4)))
    test_result_grad['pose_mean_corr'].append(abs(round(pose_corr_mean, 4)))
    test_result_grad['sum_mean_corr'].append(sum_mean_corr)

    test_result_grad_df = pd.DataFrame(test_result_grad)
    test_result_grad_df.to_csv(f'{test_result_dir}/test_result_final.csv')

    # re-initialize test directories & remove NN models for gradients
    if remove_data:
        shutil.rmtree(vector_save_dir, ignore_errors=True)
        shutil.rmtree(generated_img_dir, ignore_errors=True)
        os.makedirs(vector_save_dir, exist_ok=True)
        os.makedirs(generated_img_dir, exist_ok=True)

        for property_name in PROPERTY_NAMES:
            model_path = f'{models_dir}/stylegan_gen_vector_find_v9_nn_{property_name}.pth'
            if os.path.isfile(model_path):
                os.remove(model_path)


if __name__ == '__main__':
    ns = [4000]
    ratios = [0.2]

    for n, ratio in zip(ns, ratios):
        run_stylegan_vectorfind_v9_automated_test_final(n, ratio)
