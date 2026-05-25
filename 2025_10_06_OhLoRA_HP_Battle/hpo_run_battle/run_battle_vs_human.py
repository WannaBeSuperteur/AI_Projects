
import json
import shutil

from find_hp import (create_mock_dataset,
                     get_train_means_and_stds,
                     load_hp_optimize_model,
                     find_optimal_hps,
                     train_and_test_with_optimal_hps)
from find_hp import threshold_cutoffs

import numpy as np
import torch
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from hpo_training_model.hpo_training_model import get_valid_feature_list


NUM_CLASSES = 10
HUMAN_PREDICT_INFO = """
[ SYSTEM MESSAGE ]
데이터셋이 battle_dataset/ 경로에 저장되었습니다.
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}
 
(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)
"""


def run_human_prediction(human_trial_no, train_dataset, valid_dataset, test_dataset):
    with open('battle_dataset/hps.txt', 'r') as f:
        human_hps = json.load(f)

    print(f'[ 인간의 {human_trial_no}차 하이퍼파라미터 : {human_hps} ]')
    human_macro_f1_score = train_and_test_with_optimal_hps(dataset_name,
                                                           human_hps,
                                                           train_dataset,
                                                           valid_dataset,
                                                           test_dataset)

    return human_macro_f1_score


def save_dataset(dataset, base_path):
    os.makedirs(base_path, exist_ok=True)

    for idx in tqdm(range(len(dataset))):
        img, label = dataset[idx]
        label_str = str(int(np.argmax(label)))
        os.makedirs(os.path.join(base_path, label_str), exist_ok=True)

        file_name = f"img_{idx:04d}.png"
        save_path = os.path.join(base_path, label_str, file_name)

        if isinstance(img, torch.Tensor):
            save_image(img, save_path, normalize=True)
        elif isinstance(img, Image.Image):
            img.save(save_path)


def run_battle(dataset_name):
    print(f'\n\n[ 데이터셋 이름 : {dataset_name} ]')
    train_dataset, valid_dataset, test_dataset, hpo_model_input_data = create_mock_dataset(dataset_name)
    train_means, train_stds = get_train_means_and_stds(dataset_name, threshold_cutoffs[dataset_name])

    hp_optimize_model = load_hp_optimize_model(dataset_name)
    valid_features = get_valid_feature_list(dataset_name, threshold_cutoff=threshold_cutoffs[dataset_name])

    # 데이터셋 저장
    best_path_train = 'battle_dataset/train_dataset'
    best_path_valid = 'battle_dataset/valid_dataset'
    best_path_test = 'battle_dataset/test_dataset'

    save_dataset(train_dataset, best_path_train)
    save_dataset(valid_dataset, best_path_valid)
    save_dataset(test_dataset, best_path_test)

    # Oh-LoRA 가 먼저 예측
    print('\n\n=== Oh-LoRA 👱‍♀️ 선공 예측 ===')
    optimal_hps, pred_macro_f1_score = find_optimal_hps(hp_optimize_model,
                                                        hpo_model_input_data,
                                                        train_means,
                                                        train_stds,
                                                        valid_features,
                                                        show_result=False)

    optimal_hps = {k.replace('hp_', ''): v for k, v in optimal_hps.items()}
    optimal_hps['activation_func'] = optimal_hps.pop('actfunc')
    optimal_hps['optimizer'] = optimal_hps.pop('opt')
    optimal_hps['scheduler'] = optimal_hps.pop('sch')

    macro_f1_score = train_and_test_with_optimal_hps(dataset_name,
                                                     optimal_hps,
                                                     train_dataset,
                                                     valid_dataset,
                                                     test_dataset)

    # 인간이 예측 실시
    print('\n\n=== 인간 1차 예측 ===')
    _ = input(HUMAN_PREDICT_INFO)
    human_macro_f1_score_1 = run_human_prediction(1, train_dataset, valid_dataset, test_dataset)
    print(f'[ 인간의 Macro F1 Score = {human_macro_f1_score_1} ]')

    print('\n\n=== 인간 2차 예측 ===')
    _ = input(HUMAN_PREDICT_INFO)
    human_macro_f1_score_2 = run_human_prediction(2, train_dataset, valid_dataset, test_dataset)
    print(f'[ 인간의 Macro F1 Score = {human_macro_f1_score_2} ]')

    # 최종 결과 공개
    print(f'[ Oh-LoRA 👱‍♀️ 의 Macro F1 Score = {macro_f1_score} ] (예측: {pred_macro_f1_score})')
    print(f'[ Oh-LoRA 👱‍♀️ 의 하이퍼파라미터 = {optimal_hps} ]')

    if human_macro_f1_score_1 > macro_f1_score or human_macro_f1_score_2 > macro_f1_score:
        print(f'[ 최종 결과 : 인간 사용자의 승리 ]')
    elif human_macro_f1_score_1 == macro_f1_score or human_macro_f1_score_2 == macro_f1_score:
        print(f'[ 최종 결과 : 무승부 ]')
    else:
        print(f'[ 최종 결과 : Oh-LoRA 👱‍♀️ 의 승리 ]')

    shutil.rmtree(best_path_train)
    shutil.rmtree(best_path_valid)
    shutil.rmtree(best_path_test)


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        run_battle(dataset_name)
