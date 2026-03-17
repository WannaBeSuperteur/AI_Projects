
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
    return model


