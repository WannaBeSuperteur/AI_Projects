
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from auto_encoder import AutoEncoder_3_32_32, AutoEncoder_1_28_28
from dataset import create_dataset_df
from dataset import base_transform, AutoEncoderImageDataset

import shutil


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TEST_DIR_PATH = f'{PROJECT_DIR_PATH}/hidden_representation/test'
TEST_DIR_PATH_RECONSTRUCTION = f'{TEST_DIR_PATH}/reconstruction'
TEST_DIR_PATH_REPRESENTATION = f'{TEST_DIR_PATH}/representation'
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
# - dataset_name    (str)                      : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - ae_entire_model (nn.Module)                : Auto-Encoder 모델 전체 (Encoder + Decoder)
# - test_dataset    (torch.utils.data.Dataset) : 테스트 데이터셋

def run_reconstruction_test(dataset_name, ae_entire_model, test_dataset):
    ae_entire_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    image_save_path = os.path.join(TEST_DIR_PATH_RECONSTRUCTION, dataset_name)

    for idx, (images, _) in enumerate(test_loader):
        images = images.to(ae_entire_model.device)

        with torch.no_grad():
            decoder_outputs = ae_entire_model(images).to(torch.float32)

        for img_idx, (image, decoder_output) in enumerate(zip(images, decoder_outputs)):
            image_file_path = os.path.join(image_save_path, f'{idx}_{img_idx}_original.png')
            decoder_output_file_path = os.path.join(image_save_path, f'{idx}_{img_idx}_decoder_out.png')

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
    ae_encoder.eval()
    image_save_path = os.path.join(TEST_DIR_PATH_REPRESENTATION, dataset_name)

    # apply shuffle for test dataset, to use only few images with various labels
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
    label_and_encoder_output = []

    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(ae_encoder.device)

        with torch.no_grad():
            encoder_outputs = ae_encoder(images).to(torch.float32)

        for img_idx, (image, label, encoder_output) in enumerate(zip(images, labels, encoder_outputs)):
            encoder_output_flatten = encoder_output.detach().cpu().numpy().flatten()
            label_and_encoder_output.append({'batch_idx': idx,
                                             'img_idx': img_idx,
                                             'label': label,
                                             'encoder_output': encoder_output_flatten})

            image_file_path = os.path.join(image_save_path, f'{idx}_{img_idx}_original.png')
            image_0to1 = image / 2.0 + 0.5
            save_image(image_0to1, image_file_path, normalize=True)

        if idx >= 200:
            break

    # compute difference between encodings (= Euclidean distance)
    representation_test_result_dict = {'label_same': [], 'encoding_distance': [],
                                       'idx_i': [], 'idx_j': [],
                                       'label_i': [], 'label_j': []}
    encoding_distances_of_same_label = []
    encoding_distances_of_diff_label = []

    for i in range(len(label_and_encoder_output)):
        for j in range(i):
            idx_i = f"{label_and_encoder_output[i]['batch_idx']}_{label_and_encoder_output[i]['img_idx']}"
            idx_j = f"{label_and_encoder_output[j]['batch_idx']}_{label_and_encoder_output[j]['img_idx']}"
            label_i = int(torch.argmax(label_and_encoder_output[i]['label']))
            label_j = int(torch.argmax(label_and_encoder_output[j]['label']))
            encoder_output_i = label_and_encoder_output[i]['encoder_output']
            encoder_output_j = label_and_encoder_output[j]['encoder_output']
            encoding_distance = np.linalg.norm(encoder_output_i - encoder_output_j)

            if label_i == label_j:
                encoding_distances_of_same_label.append(encoding_distance)
            else:
                encoding_distances_of_diff_label.append(encoding_distance)

            representation_test_result_dict['label_same'].append(label_i == label_j)
            representation_test_result_dict['encoding_distance'].append(encoding_distance)
            representation_test_result_dict['idx_i'].append(idx_i)
            representation_test_result_dict['idx_j'].append(idx_j)
            representation_test_result_dict['label_i'].append(label_i)
            representation_test_result_dict['label_j'].append(label_j)

    print('save as Pandas DataFrame csv ...')
    representation_test_result_df = pd.DataFrame(representation_test_result_dict)
    representation_test_result_df.to_csv(f'{TEST_DIR_PATH_REPRESENTATION}/test_result_{dataset_name}.csv')

    # print statstics
    print(f'\nSAME LABEL - distance mean : {np.mean(encoding_distances_of_same_label)}')
    print(f'SAME LABEL - distance std : {np.std(encoding_distances_of_same_label)}')
    print(f'SAME LABEL - distance max : {np.max(encoding_distances_of_same_label)}')
    print(f'SAME LABEL - distance min : {np.min(encoding_distances_of_same_label)}')

    print(f'\nDIFFERENT LABEL - distance mean : {np.mean(encoding_distances_of_diff_label)}')
    print(f'DIFFERENT LABEL - distance std : {np.std(encoding_distances_of_diff_label)}')
    print(f'DIFFERENT LABEL - distance max : {np.max(encoding_distances_of_diff_label)}')
    print(f'DIFFERENT LABEL - distance min : {np.min(encoding_distances_of_diff_label)}')

    # save 10 shortest-distance image pairs
    representation_test_result_df.sort_values(by=['encoding_distance'], inplace=True)
    print('')

    shortest_distance_image_copy_path = os.path.join(image_save_path, 'most_similar_image_pairs')
    os.makedirs(shortest_distance_image_copy_path, exist_ok=True)

    for i in range(25):
        idx_i = representation_test_result_df.iloc[i]['idx_i']
        idx_j = representation_test_result_df.iloc[i]['idx_j']
        label_i = representation_test_result_df.iloc[i]['label_i']
        label_j = representation_test_result_df.iloc[i]['label_j']
        encoding_distance = representation_test_result_df.iloc[i]['encoding_distance']

        print(f'shortest distance ({i}): index {idx_i} (label: {label_i}) and {idx_j} (label: {label_j}), '
              f'encoding distance: {encoding_distance}')

        # copy image
        image_saved_path_i = os.path.join(image_save_path, f'{idx_i}_original.png')
        image_saved_path_j = os.path.join(image_save_path, f'{idx_j}_original.png')
        image_copy_path_i = os.path.join(shortest_distance_image_copy_path, f'shortest_dist_{i+1}-th_idx_{idx_i}.png')
        image_copy_path_j = os.path.join(shortest_distance_image_copy_path, f'shortest_dist_{i+1}-th_idx_{idx_j}.png')

        shutil.copy(image_saved_path_i, image_copy_path_i)
        shutil.copy(image_saved_path_j, image_copy_path_j)


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
    os.makedirs(TEST_DIR_PATH_REPRESENTATION, exist_ok=True)
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        os.makedirs(f'{TEST_DIR_PATH_RECONSTRUCTION}/{dataset_name}', exist_ok=True)
        os.makedirs(f'{TEST_DIR_PATH_REPRESENTATION}/{dataset_name}', exist_ok=True)
        print(f'\n==== DATASET: {dataset_name} ====\n')

        test_dataset = load_test_dataset(dataset_name)

        ae_encoder = load_ae_encoder(dataset_name)
        ae_entire_model = load_ae_entire_model(dataset_name)

        print('reconstruction test ...')
        run_reconstruction_test(dataset_name, ae_entire_model, test_dataset)

        print('hidden representation test ...')
        run_hidden_representation_test(dataset_name, ae_encoder, test_dataset)
