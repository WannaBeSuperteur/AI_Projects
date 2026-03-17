
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from auto_encoder import AutoEncoder_3_32_32, AutoEncoder_1_28_28
from dataset import create_dataset_df
from dataset import base_transform, AutoEncoderImageDataset


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TEST_DIR_PATH = f'{PROJECT_DIR_PATH}/hidden_representation/test'
TEST_DIR_PATH_RECONSTRUCTION = f'{TEST_DIR_PATH}/reconstruction'
TEST_BATCH_SIZE = 4


# Auto-Encoder 모델의 Encoder 불러오기
# Create Date : 2026.03.17
# Last Update Date : 2026.03.18
# - model mapping to device

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - ae_encoder (nn.Module) : load 된 Auto-Encoder 모델의 Encoder

def load_ae_encoder(dataset_name):

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for loading model : {device}')

    if dataset_name == 'cifar_10':
        model = AutoEncoder_3_32_32().encoder
    else:
        model = AutoEncoder_1_28_28().encoder

    model_path = f'{PROJECT_DIR_PATH}/models/ae_encoder_{dataset_name}.pt'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.device = device
    model.to(device)

    return model


# Auto-Encoder 모델 전체 불러오기
# Create Date : 2026.03.17
# Last Update Date : 2026.03.18
# - model mapping to device

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
    model.to(device)

    return model


# Auto-Encoder 모델의 Re-construction 성능 테스트 (시각적 결과)
# Create Date : 2026.03.18
# Last Update Date : -

# Arguments:
# - ae_entire_model (nn.Module)                : Auto-Encoder 모델 전체 (Encoder + Decoder)
# - test_dataset    (torch.utils.data.Dataset) : 테스트 데이터셋

def run_reconstruction_test(ae_entire_model, test_dataset):
    ae_entire_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    for idx, (images, _) in enumerate(test_loader):
        images = images.to(ae_entire_model.device)

        with torch.no_grad():
            decoder_outputs = ae_entire_model(images).to(torch.float32)

        for img_idx, (image, decoder_output) in enumerate(zip(images, decoder_outputs)):
            image_file_path = os.path.join(TEST_DIR_PATH_RECONSTRUCTION, f'{idx}_{img_idx}_original.png')
            decoder_output_file_path = os.path.join(TEST_DIR_PATH_RECONSTRUCTION, f'{idx}_{img_idx}_decoder_out.png')

            # -1 to 1 -> 0 to 1
            image_0to1 = image / 2.0 + 0.5
            decoder_output_0to1 = decoder_output / 2.0 + 0.5

            save_image(image_0to1, image_file_path, normalize=True)
            save_image(decoder_output_0to1, decoder_output_file_path, normalize=True)

        if idx >= 30:
            break


# Auto-Encoder 모델의 Hidden Representation 테스트 (가까운 이미지 & 거리가 큰 이미지)
# Create Date : 2026.03.18
# Last Update Date : -

# Arguments:
# - dataset_name (str)                      : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - ae_encoder   (nn.Module)                : Auto-Encoder 모델의 Encoder
# - test_dataset (torch.utils.data.Dataset) : 테스트 데이터셋

def run_hidden_representation_test(dataset_name, ae_encoder, test_dataset):
    raise NotImplementedError


# Auto Encoder 테스트용 데이터셋 로딩
# Create Date : 2026.03.17
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - test_dataset (torch.utils.data.Dataset) : 테스트 데이터셋

def load_test_dataset(dataset_name):
    test_dataset_df = create_dataset_df(dataset_name, tvt_type='test')
    test_dataset = AutoEncoderImageDataset(test_dataset_df,
                                           transform=base_transform,
                                           dataset_name=dataset_name,
                                           tvt_type='test')

    return test_dataset


if __name__ == '__main__':
    os.makedirs(TEST_DIR_PATH, exist_ok=True)
    os.makedirs(TEST_DIR_PATH_RECONSTRUCTION, exist_ok=True)
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        print(f'\n==== DATASET: {dataset_name} ====\n')

        test_dataset = load_test_dataset(dataset_name)
        print(f'test dataset : {test_dataset}')

        ae_encoder = load_ae_encoder(dataset_name)
        ae_entire_model = load_ae_entire_model(dataset_name)

        run_reconstruction_test(ae_entire_model, test_dataset)
        run_hidden_representation_test(dataset_name, ae_encoder, test_dataset)
