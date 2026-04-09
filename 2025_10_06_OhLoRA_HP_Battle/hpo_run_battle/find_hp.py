

# 하이퍼파라미터 탐색 가능한 모의 데이터셋 생성
# Create Date : 2026.04.09
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('mnist', 'fashion_mnist' or 'cifar_10')

# Returns:
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - valid_dataset (torch.utils.data.Dataset) : 검증 (valid) 데이터셋
# - test_dataset  (torch.utils.data.Dataset) : 테스트 데이터셋

def create_mock_dataset(dataset_name):
    raise NotImplementedError


# 기 학습된 최적 하이퍼파라미터 탐색 모델 로딩
# Create Date : 2026.04.09
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('mnist', 'fashion_mnist' or 'cifar_10')

# Returns:
# - hp_optimize_model (torch.nn.module) : 기 학습된 최적 하이퍼파라미터 탐색 모델

def load_hp_optimize_model(dataset_name):
    raise NotImplementedError


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
        train_dataset, valid_dataset, test_dataset = create_mock_dataset(dataset_name)
        hp_optimize_model = load_hp_optimize_model(dataset_name)
        optimal_hps = find_optimal_hps(hp_optimize_model, train_dataset)
        macro_f1_score = train_and_test_with_optimal_hps(optimal_hps, train_dataset, valid_dataset, test_dataset)

        print(f'dataset_name : {dataset_name}')
        print(f'optimal Hyper-params: {optimal_hps}')
        print(f'Macro F1 Score: {macro_f1_score}')
