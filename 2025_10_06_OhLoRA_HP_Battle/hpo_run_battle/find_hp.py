
import numpy as np
import torch
import random

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from hpo_training_data.create_hpo_model_train_data import generate_constraints
from hpo_training_data.train_cnn import load_dataset, encode_train_dataset
from hpo_training_data.train_cnn import EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA, NUM_CLASSES
from hidden_representation.auto_encoder import AutoEncoderEncoder_1_28_28, AutoEncoderEncoder_3_32_32
from hpo_training_model.hpo_training_model import (load_trained_hpo_model,
                                                   get_valid_feature_list,
                                                   get_means_and_stds,
                                                   merge_dataset_df)
from hpo_training_model.hpo_training_model import NUM_FEATURES_OUTPUT


threshold_cutoffs = {'cifar_10': 0.2, 'fashion_mnist': 0.175, 'mnist': 0.35}
HP_RANDOM_INIT_COUNT = 10


def convert_to_train_data(ae_encoder, train_dataset, train_dataset_label_distrib, labels_trained):

    # get train images embedding
    encoding_result = encode_train_dataset(ae_encoder, train_dataset)
    encoding_mean = np.mean(encoding_result, axis=0)
    encoding_std = np.std(encoding_result, axis=0)

    # label distribution
    total_train_images = sum(train_dataset_label_distrib)
    max_min_of_labels = max(train_dataset_label_distrib) / max(1, min(train_dataset_label_distrib))
    avg_std_of_labels = np.mean(train_dataset_label_distrib) / max(0.001, np.std(train_dataset_label_distrib))
    largest_label_percentage = train_dataset_label_distrib[0] / total_train_images

    # input and output data
    hpo_model_input_data = {'encoding_mean': encoding_mean,
                            'encoding_std': encoding_std,
                            'total_train_images': total_train_images,
                            'max_min_of_labels': max_min_of_labels,
                            'avg_std_of_labels': avg_std_of_labels,
                            'largest_label_percentage': largest_label_percentage,
                            'labels_trained': labels_trained}

    return hpo_model_input_data


def get_train_means_and_stds(dataset_name, threshold_cutoff):
    valid_features = get_valid_feature_list(dataset_name, threshold_cutoff=threshold_cutoff)
    merged_dataset_df = merge_dataset_df(dataset_name, valid_features=valid_features)
    merged_dataset_size = len(merged_dataset_df)
    merged_dataset_train_size = int(0.9 * merged_dataset_size)

    train_df_raw = merged_dataset_df.iloc[:merged_dataset_train_size, :]
    train_means, train_stds = get_means_and_stds(train_df_raw)

    return train_means, train_stds


def init_hps(all_hps_list):
    hps_dict = {}
    if 'hp_dropout_conv_earlier' in all_hps_list:
        hps_dict['hp_dropout_conv_earlier'] = random.random() * 0.3
    if 'hp_dropout_conv_later' in all_hps_list:
        hps_dict['hp_dropout_conv_later'] = random.random() * 0.3
    if 'hp_dropout_fc' in all_hps_list:
        hps_dict['hp_dropout_fc'] = random.random() * 0.6
    if 'hp_lr' in all_hps_list:
        hps_dict['hp_lr'] = pow(10, random.random() * 2.45 - 4.7)

    actfunc_one_hot_features = list(filter(lambda x: x.startswith('actfunc'), all_hps_list))
    opt_one_hot_features = list(filter(lambda x: x.startswith('opt'), all_hps_list))
    sch_one_hot_features = list(filter(lambda x: x.startswith('sch'), all_hps_list))

    if len(actfunc_one_hot_features) >= 1:
        hps_dict['actfunc'] = '_'.join(random.choice(actfunc_one_hot_features).split('_')[1:])

    if len(opt_one_hot_features) >= 1:
        hps_dict['opt'] = '_'.join(random.choice(opt_one_hot_features).split('_')[1:])

    if len(sch_one_hot_features) >= 1:
        hps_dict['sch'] = '_'.join(random.choice(sch_one_hot_features).split('_')[1:])

    return hps_dict



# 하이퍼파라미터 탐색 가능한 모의 데이터셋 생성
# Create Date : 2026.04.09
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('mnist', 'fashion_mnist' or 'cifar_10')

# Returns:
# - train_dataset        (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - valid_dataset        (torch.utils.data.Dataset) : 검증 (valid) 데이터셋
# - test_dataset         (torch.utils.data.Dataset) : 테스트 데이터셋
# - hpo_model_input_data (dict)                     : 기 학습된 하이퍼파라미터 최적화 모델의 입력 데이터

def create_mock_dataset(dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load Auto-Encoder Encoder
    if dataset_name == 'cifar_10':
        ae_encoder = AutoEncoderEncoder_3_32_32()
    else:
        ae_encoder = AutoEncoderEncoder_1_28_28()

    model_path = f'{PROJECT_DIR_PATH}/models/ae_encoder_{dataset_name}.pt'
    ae_encoder_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    ae_encoder.load_state_dict(ae_encoder_state_dict)
    ae_encoder.to(device)
    ae_encoder.device = device

    while True:

        # load constraints
        constraints = generate_constraints(dataset_name)

        # load dataset & check label distribution condition
        train_dataset, valid_dataset, test_dataset, train_dataset_label_distrib, test_dataset_label_distrib, labels_trained = (
            load_dataset(dataset_name, constraints, dataset_path_prefix='../hpo_training_data/'))

        too_many_train_data = sum(train_dataset_label_distrib) > 1500
        too_few_classes = len(train_dataset_label_distrib) < 5 or len(test_dataset_label_distrib) < 5
        too_few_minor_class_data = too_few_classes or train_dataset_label_distrib[4] < 125 or test_dataset_label_distrib[4] < 25

        # accept / reject dataset using data label distribution
        if too_many_train_data or too_few_minor_class_data:
            print(f'rejected distribution: train={train_dataset_label_distrib}, test={test_dataset_label_distrib}')
            continue

        # convert to train data
        hpo_model_input_data = convert_to_train_data(ae_encoder=ae_encoder,
                                                     train_dataset=train_dataset,
                                                     train_dataset_label_distrib=train_dataset_label_distrib,
                                                     labels_trained=labels_trained)

        return train_dataset, valid_dataset, test_dataset, hpo_model_input_data


# 기 학습된 최적 하이퍼파라미터 탐색 모델 로딩
# Create Date : 2026.04.09
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('mnist', 'fashion_mnist' or 'cifar_10')

# Returns:
# - hp_optimize_model (torch.nn.module) : 기 학습된 최적 하이퍼파라미터 탐색 모델

def load_hp_optimize_model(dataset_name):
    valid_features = get_valid_feature_list(dataset_name, threshold_cutoff=threshold_cutoffs[dataset_name])
    num_input_features = len(valid_features) - NUM_FEATURES_OUTPUT

    hp_optimize_model = load_trained_hpo_model(num_input_features, dataset_name)
    return hp_optimize_model


# 기 학습된 최적 하이퍼파라미터 탐색 모델을 이용한 최적 하이퍼파라미터 탐색 (hill-climbing 방식)
# Create Date : 2026.04.12
# Last Update Date : -

# Arguments:
# - hp_optimize_model    (torch.nn.module) : 기 학습된 최적 하이퍼파라미터 탐색 모델
# - hpo_model_input_data (dict)            : 기 학습된 하이퍼파라미터 최적화 모델의 입력 데이터
# - train_means          (dict(float))     : 학습 데이터셋의 각 input column 별 평균값 목록
# - train_stds           (dict(float))     : 학습 데이터셋의 각 input column 별 표준편차 목록
# - valid_features       (list)            : HPO 모델 학습용 데이터셋의 valid feature (= column) 리스트

# Returns:
# - optimal_hps (dict) : 학습된 탐색 모델 + hill-climbing 결과에 의한 최적 하이퍼파라미터 목록

def find_optimal_hps(hp_optimize_model, hpo_model_input_data, train_means, train_stds, valid_features):
    base_input_data = []
    all_hps_list = []

    dataset_stat_features = ['total_train_images',
                             'max_min_of_labels',
                             'avg_std_of_labels',
                             'largest_label_percentage']

    # dataset stat
    for dataset_stat_feature in dataset_stat_features:
        if dataset_stat_feature in valid_features:
            base_input_data.append(hpo_model_input_data[dataset_stat_feature])

    # hyper-params (1)
    hps_1 = ['dropout_conv_earlier', 'dropout_conv_later', 'dropout_fc', 'lr']
    for hp in hps_1:
        if f'hp_{hp}' in valid_features:
            all_hps_list.append(f'hp_{hp}')
            base_input_data.append({f'hp_{hp}': None})

    # encoding mean and std
    for i in range(EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA):
        if f'encoding_mean_{i}' in valid_features:
            base_input_data.append(hpo_model_input_data[f'encoding_mean'][i])
        if f'encoding_std_{i}' in valid_features:
            base_input_data.append(hpo_model_input_data[f'encoding_std'][i])

    # labels trained or not
    for i in range(NUM_CLASSES):
        if f'labels_trained_{i}' in valid_features:
            base_input_data.append(hpo_model_input_data[f'labels_trained_{i}'])

    # hyper-params (2)
    hps_2 = ['actfunc_relu', 'actfunc_leaky_relu',
             'opt_adam', 'opt_adamw',
             'sch_exp_90', 'sch_exp_95', 'sch_exp_98', 'sch_cosine']

    for hp in hps_2:
        if hp in valid_features:
            all_hps_list.append(hp.split('_')[0])
            base_input_data.append({hp: None})

    print(base_input_data)
    print(all_hps_list)

    # find best hyper-param
    best_hps = {}

    for i in range(HP_RANDOM_INIT_COUNT):
        while True:
            current_hps = init_hps(all_hps_list)
            print(current_hps)
            break

    raise NotImplementedError


# 탐색한 최적 하이퍼파라미터를 이용한 학습 시의 Macro F1 Score 측정
# Create Date : 2026.04.10
# Last Update Date : -

# Arguments:
# - optimal_hps   (dict)                     : 학습된 탐색 모델 + hill-climbing 결과에 의한 최적 하이퍼파라미터 목록
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - valid_dataset (torch.utils.data.Dataset) : 검증 (valid) 데이터셋
# - test_dataset  (torch.utils.data.Dataset) : 테스트 데이터셋

def train_and_test_with_optimal_hps(optimal_hps, train_dataset, valid_dataset, test_dataset):
    raise NotImplementedError


if __name__ == '__main__':
    dataset_names = ['mnist', 'fashion_mnist', 'cifar_10']

    # run baseline CNN training & test for each dataset
    for dataset_name in dataset_names:
        train_dataset, valid_dataset, test_dataset, hpo_model_input_data = create_mock_dataset(dataset_name)
        train_means, train_stds = get_train_means_and_stds(dataset_name, threshold_cutoffs[dataset_name])

        hp_optimize_model = load_hp_optimize_model(dataset_name)
        valid_features = get_valid_feature_list(dataset_name, threshold_cutoff=threshold_cutoffs[dataset_name])

        optimal_hps = find_optimal_hps(hp_optimize_model, hpo_model_input_data, train_means, train_stds, valid_features)
        macro_f1_score = train_and_test_with_optimal_hps(optimal_hps, train_dataset, valid_dataset, test_dataset)

        print(f'dataset_name : {dataset_name}')
        print(f'optimal Hyper-params: {optimal_hps}')
        print(f'Macro F1 Score: {macro_f1_score}')
