
from auto_encoder import AutoEncoder_1_28_28, AutoEncoder_3_32_32
from dataset import split_into_train_and_valid, create_dataset_df


# Auto Encoder 모델 로딩 (학습 전의 모델 로딩 -> 이후 학습 실시)
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - ae_model (torch.nn.modules) : Auto-Encoder 모델

def load_ae_model_before_train(dataset_name):
    if dataset_name == 'cifar_10':
        ae_model = AutoEncoder_3_32_32()
    else:
        ae_model = AutoEncoder_1_28_28()

    return ae_model


# Auto Encoder 학습용 데이터셋 로딩
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - test_dataset  (torch.utils.data.Dataset) : 테스트 데이터셋

def load_dataset(dataset_name):
    train_dataset_df = create_dataset_df(dataset_name, tvt_type='train')
    test_dataset_df = create_dataset_df(dataset_name, tvt_type='test')

    raise NotImplementedError


# Auto Encoder 학습 실시 및 모델 저장
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋

def train_ae(train_dataset):
    raise NotImplementedError


# Auto Encoder 테스트 실시 및 테스트 결과 저장
# Create Date : 2026.03.16
# Last Update Date : -

# Arguments:
# - test_dataset (torch.utils.data.Dataset) : 테스트 데이터

def test_ae(test_dataset):
    raise NotImplementedError


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        ae_model = load_ae_model_before_train(dataset_name)
        train_dataset, test_dataset = load_dataset(dataset_name)
        train_train_dataset, train_valid_dataset = split_into_train_and_valid(train_dataset)

        train_ae(train_dataset)
        test_ae(test_dataset)
