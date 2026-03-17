
import os
import torch

from auto_encoder import AutoEncoder_3_32_32, AutoEncoder_1_28_28


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Auto-Encoder 모델의 Encoder 불러오기
# Create Date : 2026.03.17
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - ae_encoder (nn.Module) : load 된 Auto-Encoder 모델의 Encoder

def load_ae_encoder(dataset_name):

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for loading model : {device}')

    if dataset_name == 'cifar_10':
        model = AutoEncoder_3_32_32()
    else:
        model = AutoEncoder_1_28_28()

    model_path = f'{PROJECT_DIR_PATH}/models/ae_encoder_{dataset_name}.pt'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.device = device
    return model


# Auto-Encoder 모델 전체 불러오기
# Create Date : 2026.03.17
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - ae_entire_model (nn.Module) : load 된 Auto-Encoder 모델 전체 (Encoder + Decoder)

def load_ae_entire_model(dataset_name):

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for loading model : {device}')

    if dataset_name == 'cifar_10':
        model = AutoEncoder_3_32_32()
    else:
        model = AutoEncoder_1_28_28()

    model_path = f'{PROJECT_DIR_PATH}/models/ae_model_{dataset_name}.pt'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.device = device
    return model


# Auto-Encoder 모델의 Re-construction 성능 테스트 (시각적 결과)
# Create Date : 2026.03.18
# Last Update Date : -

# Arguments:
# - dataset_name    (str)       : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - ae_entire_model (nn.Module) : Auto-Encoder 모델 전체 (Encoder + Decoder)

def run_reconstruction_test(dataset_name, ae_entire_model):
    raise NotImplementedError


# Auto-Encoder 모델의 Hidden Representation 테스트 (가까운 이미지 & 거리가 큰 이미지)
# Create Date : 2026.03.18
# Last Update Date : -

# Arguments:
# - dataset_name (str)       : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - ae_encoder   (nn.Module) : Auto-Encoder 모델의 Encoder

def run_hidden_representation_test(dataset_name, ae_encoder):
    raise NotImplementedError


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        print(f'\n==== DATASET: {dataset_name} ====\n')

        ae_encoder = load_ae_encoder(dataset_name)
        ae_entire_model = load_ae_encoder(dataset_name)

        run_reconstruction_test(dataset_name, ae_entire_model)
        run_hidden_representation_test(dataset_name, ae_encoder)
