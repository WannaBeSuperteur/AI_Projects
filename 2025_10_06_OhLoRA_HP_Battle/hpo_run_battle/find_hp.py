
import numpy as np
import torch

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from hpo_training_data.create_hpo_model_train_data import generate_constraints
from hpo_training_data.train_cnn import load_dataset, encode_train_dataset
from hidden_representation.auto_encoder import AutoEncoderEncoder_1_28_28, AutoEncoderEncoder_3_32_32
from hpo_training_model.hpo_training_model import load_trained_hpo_model, get_valid_feature_list
from hpo_training_model.hpo_training_model import NUM_FEATURES_OUTPUT


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
    threshold_cutoffs = {'cifar_10': 0.2, 'fashion_mnist': 0.175, 'mnist': 0.35}

    valid_features = get_valid_feature_list(dataset_name, threshold_cutoff=threshold_cutoffs[dataset_name])
    num_input_features = len(valid_features) - NUM_FEATURES_OUTPUT

    hp_optimize_model = load_trained_hpo_model(num_input_features, dataset_name)
    return hp_optimize_model


# 기 학습된 최적 하이퍼파라미터 탐색 모델을 이용한 최적 하이퍼파라미터 탐색 (hill-climbing 방식)
# Create Date : 2026.04.09
# Last Update Date : -

# Arguments:
# - hp_optimize_model (torch.nn.module)          : 기 학습된 최적 하이퍼파라미터 탐색 모델
# - train_dataset     (torch.utils.data.Dataset) : 학습 (train) 데이터셋

# Returns:
# - optimal_hps (dict) : 학습된 탐색 모델 + hill-climbing 결과에 의한 최적 하이퍼파라미터 목록

def find_optimal_hps(hp_optimize_model, train_dataset):
    raise NotImplementedError


# 탐색한 최적 하이퍼파라미터를 이용한 학습 시의 Macro F1 Score 측정
# Create Date : 2026.04.09
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
        hp_optimize_model = load_hp_optimize_model(dataset_name)
        optimal_hps = find_optimal_hps(hp_optimize_model, train_dataset)
        macro_f1_score = train_and_test_with_optimal_hps(optimal_hps, train_dataset, valid_dataset, test_dataset)

        print(f'dataset_name : {dataset_name}')
        print(f'optimal Hyper-params: {optimal_hps}')
        print(f'Macro F1 Score: {macro_f1_score}')
