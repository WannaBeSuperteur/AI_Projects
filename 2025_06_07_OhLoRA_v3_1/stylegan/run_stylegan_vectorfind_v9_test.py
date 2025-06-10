try:
    from stylegan_vectorfind_v9.main import main_svm as stylegan_vectorfind_v9_main_svm
    from stylegan_vectorfind_v9.main import main_gradient as stylegan_vectorfind_v9_main_gradient
    import stylegan_common.stylegan_generator as gen

    from common import (load_existing_stylegan_finetune_v9,
                        load_existing_stylegan_vectorfind_v9,
                        load_merged_property_score_cnn)

except:
    from stylegan.stylegan_vectorfind_v9.main import main_svm as stylegan_vectorfind_v9_main_svm
    from stylegan.stylegan_vectorfind_v9.main import main_gradient as stylegan_vectorfind_v9_main_gradient
    import stylegan.stylegan_common.stylegan_generator as gen

    from stylegan.common import (load_existing_stylegan_finetune_v9,
                                 load_existing_stylegan_vectorfind_v9,
                                 load_merged_property_score_cnn)

from run_stylegan_vectorfind_v9_svm import get_property_change_vectors
from run_stylegan_vectorfind_v9_svm import run_image_generation_test as run_image_generation_test_svm
from run_stylegan_vectorfind_v9_svm import run_property_score_compare_test as run_property_score_compare_test_svm

from run_stylegan_vectorfind_v9_gradient import get_property_change_gradient_nn
from run_stylegan_vectorfind_v9_gradient import run_image_generation_gradient as run_image_generation_test_gradient
from run_stylegan_vectorfind_v9_gradient import run_property_score_compare_gradient as run_property_score_compare_test_gradient

import torch
import pandas as pd

import os
import shutil
import time

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256
finetune_v9_generator = None

test_result_svm = {'n': [], 'k': [], 'time': [],
                   'svm_eyes_acc': [], 'svm_mouth_acc': [], 'svm_pose_acc': [],
                   'eyes_mean_corr': [], 'mouth_mean_corr': [], 'pose_mean_corr': [], 'sum_mean_corr': []}

test_result_grad = {'n': [], 'time': [],
                    'nn_eyes_mse': [], 'nn_mouth_mse': [], 'nn_pose_mse': [],
                    'eyes_mean_corr': [], 'mouth_mean_corr': [], 'pose_mean_corr': [], 'sum_mean_corr': []}

image_gen_report_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/image_generation_report'
vector_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/property_score_vectors'
generated_img_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/inference_test_after_training'
test_result_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/test_result'
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
# Last Update Date : -

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
    shutil.rmtree(image_gen_report_dir)
    shutil.rmtree(vector_save_dir)
    shutil.rmtree(generated_img_dir)

    os.makedirs(image_gen_report_dir)
    os.makedirs(vector_save_dir)
    os.makedirs(generated_img_dir)


if __name__ == '__main__':
    ns = [4000, 8000, 20000, 40000, 80000]
    ratios = [0.2, 0.2, 0.2, 0.2, 0.2]
    layer_name = 'w'

    for n, ratio in zip(ns, ratios):
        run_stylegan_vectorfind_v9_automated_test_gradient(n, layer_name)
