import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchinfo import summary

try:
    from common import save_model_structure_pdf
    from stylegan_vectorfind_v9.nn_train_utils import (run_train_process,
                                                       run_test_process,
                                                       get_mid_vector_dim,
                                                       create_dataloader)
except:
    from stylegan.common import save_model_structure_pdf
    from stylegan.stylegan_vectorfind_v9.nn_train_utils import (run_train_process,
                                                                run_test_process,
                                                                get_mid_vector_dim,
                                                                create_dataloader)

import time
import os
import sys

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(PROJECT_DIR_PATH)

try:
    import stylegan_common.stylegan_generator_inference as infer
    from common import load_merged_property_score_cnn
except:
    import stylegan.stylegan_common.stylegan_generator_inference as infer
    from stylegan.common import load_merged_property_score_cnn

# remove warnings
import warnings
warnings.filterwarnings('ignore')


ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINAL_HIDDEN_DIMS_W = 512
HIDDEN_DIMS_MAPPING_SPLIT1 = 512 + 2048
HIDDEN_DIMS_MAPPING_SPLIT2 = 512 + 512

ORIGINALLY_PROPERTY_DIMS = 7    # 원래 property (eyes, hair_color, hair_length, mouth, pose,
                                #               background_mean, background_std) 목적으로 사용된 dimension 값
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
IMG_GENERATION_BATCH_SIZE = 20

PROPERTY_NAMES = ['eyes', 'mouth', 'pose']


class SimpleNNForVectorFindV9(nn.Module):
    def __init__(self, mid_vector_dim):
        super(SimpleNNForVectorFindV9, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(mid_vector_dim, 768 + mid_vector_dim // 4),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768 + mid_vector_dim // 4, 128 + mid_vector_dim // 8),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.fc_final = nn.Linear(128 + mid_vector_dim // 8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_final(x)
        return x


class DatasetForSimpleNNForVectorFindV9(Dataset):
    def __init__(self, dataset_arr, mid_vector_dim):
        self.dataset_arr = dataset_arr
        self.mid_vector_dim = mid_vector_dim

    def __len__(self):
        return len(self.dataset_arr)

    def __getitem__(self, idx):
        input_data = self.dataset_arr[idx][:self.mid_vector_dim]
        output_data = self.dataset_arr[idx][self.mid_vector_dim:]
        return input_data, output_data


# intermediate vector 샘플링 및 해당 vector 값으로 생성된 이미지에 대한 property score 계산
# Create Date : 2025.06.10
# Last Update Date : -

# Arguments:
# - finetune_v9_generator (nn.Module) : StyleGAN-FineTune-v9 의 Generator
# - property_score_cnn    (nn.Module) : 핵심 속성 값 계산용 CNN 모델
# - layer_name            (str)       : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름
#                                       ('mapping_split1', 'mapping_split2' or 'w')
# - n                     (int)       : sampling 할 intermediate vector 의 개수

# Returns:
# - mid_vectors_all      (NumPy array)       : sampling 된 intermediate vector
# - property_scores_dict (dict(NumPy array)) : sampling 된 intermediate vector 생성 이미지의 Pre-trained CNN 도출 핵심 속성값
#                                              {'eyes': NumPy array, 'mouth': NumPy array, 'pose': NumPy array}

def sample_vector_and_compute_property_scores(finetune_v9_generator, property_score_cnn, layer_name, n=4000):
    save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v9/inference_test_during_training'

    z = np.random.normal(0, 1, size=(n, ORIGINAL_HIDDEN_DIMS_Z)).astype(np.float64)
    property_scores_dict = {'eyes': np.zeros((n, 1)), 'mouth': np.zeros((n, 1)), 'pose': np.zeros((n, 1))}

    if layer_name == 'w':
        mid_vectors_all = np.zeros((n, ORIGINAL_HIDDEN_DIMS_W)).astype(np.float64)
    elif layer_name == 'mapping_split1':
        mid_vectors_all = np.zeros((n, HIDDEN_DIMS_MAPPING_SPLIT1)).astype(np.float64)
    else:  # mapping_split2
        mid_vectors_all = np.zeros((n, HIDDEN_DIMS_MAPPING_SPLIT2)).astype(np.float64)

    additional = np.random.normal(0, 1, size=(n, ORIGINALLY_PROPERTY_DIMS)).astype(np.float64)

    for i in range(n // IMG_GENERATION_BATCH_SIZE):
        if i % 10 == 0:
            print(f'synthesizing for batch {i} ...')

        z_ = z[i * IMG_GENERATION_BATCH_SIZE : (i+1) * IMG_GENERATION_BATCH_SIZE]
        additional_ = additional[i * IMG_GENERATION_BATCH_SIZE : (i+1) * IMG_GENERATION_BATCH_SIZE]

        images, mid_vectors = infer.synthesize(finetune_v9_generator,
                                               num=IMG_GENERATION_BATCH_SIZE,
                                               save_dir=save_dir,
                                               z=z_,
                                               label=additional_,
                                               img_name_start_idx=0,
                                               verbose=False,
                                               save_img=False,
                                               return_img=True,
                                               return_vector_at=layer_name)

        mid_vectors_all[i * IMG_GENERATION_BATCH_SIZE : (i+1) * IMG_GENERATION_BATCH_SIZE] = mid_vectors

        with torch.no_grad():
            for image_no in range(IMG_GENERATION_BATCH_SIZE):
                image = images[image_no]
                image_ = image / 255.0
                image_ = (image_ - 0.5) / 0.5
                image_ = torch.tensor(image_).type(torch.float32)
                image_ = image_.permute(2, 0, 1)

                property_scores = property_score_cnn(image_.unsqueeze(0).cuda())
                property_scores_np = property_scores.detach().cpu().numpy()
                property_scores_dict['eyes'][i * IMG_GENERATION_BATCH_SIZE + image_no][0] = property_scores_np[0][0]
                property_scores_dict['mouth'][i * IMG_GENERATION_BATCH_SIZE + image_no][0] = property_scores_np[0][3]
                property_scores_dict['pose'][i * IMG_GENERATION_BATCH_SIZE + image_no][0] = property_scores_np[0][4]

    return mid_vectors_all, property_scores_dict


# StyleGAN-VectorFind-v9 모델의 Gradient (= 핵심 속성 값 변화 벡터) 탐색 용 간단한 딥러닝 모델 정의
# Create Date : 2025.06.10
# Last Update Date : 2025.06.11
# - 변수명 수정 (vectorfind_v9_gradient_nn -> vectorfind_v9_nn)

# Arguments:
# - layer_name (str) : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름 ('mapping_split1', 'mapping_split2' or 'w')

# Returns:
# - vectorfind_v9_nn (nn.Module) : StyleGAN-VectorFind-v9 Gradient (= 핵심 속성 값 변화 벡터) 탐색 용 딥러닝 모델

def define_nn_model(layer_name):
    mid_vector_dim = get_mid_vector_dim(layer_name)

    vectorfind_v9_nn = SimpleNNForVectorFindV9(mid_vector_dim)
    vectorfind_v9_nn.optimizer = torch.optim.AdamW(vectorfind_v9_nn.parameters(), lr=0.001)
    vectorfind_v9_nn.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=vectorfind_v9_nn.optimizer,
                                                                        gamma=0.95)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vectorfind_v9_nn.to(device)
    vectorfind_v9_nn.device = device

    summary(vectorfind_v9_nn, input_size=(TRAIN_BATCH_SIZE, mid_vector_dim))
    return vectorfind_v9_nn


# StyleGAN-VectorFind-v9 핵심 속성 값 변화 벡터 (= Gradient) 추출 용 Neural Network 의 구조를 PDF 파일로 저장
# Create Date : 2025.06.11
# Last Update Date : -

# Arguments:
# - gradient_nn (nn.Module) : StyleGAN-VectorFind-v9 핵심 속성 값 변화 벡터 (= Gradient) 탐색 용 딥러닝 모델
# - layer_name  (str)       : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름
#                             ('mapping_split1', 'mapping_split2' or 'w')

# Returns:
# - stylegan/model_structure_pdf 에 해당 Neural Network 의 구조를 나타내는 PDF 파일 저장

def create_gradient_nn_structure_pdf(gradient_nn, layer_name):
    if layer_name == 'w':
        nn_input_dims = 512
    elif layer_name == 'mapping_split1':
        nn_input_dims = 512 + 2048
    else:  # mapping_split2
        nn_input_dims = 512 + 512

    save_model_structure_pdf(gradient_nn,
                             model_name=f'vectorfind_v9_gradient_nn_{layer_name}',
                             input_size=(TRAIN_BATCH_SIZE, nn_input_dims),
                             print_frozen=False)


# StyleGAN-FineTune-v9 모델을 이용한 vector find 실시 (간단한 딥러닝 & Gradient 이용)
# Create Date : 2025.06.10
# Last Update Date : 2025.06.12
# - 일부 property name 에 대한 학습 지원

# Arguments:
# - finetune_v9_generator (nn.Module) : StyleGAN-FineTune-v9 의 Generator
# - device                (device)    : Property Score CNN 로딩을 위한 device (GPU 등)
# - n                     (int)       : 총 생성할 이미지 sample 개수
# - layer_name            (str)       : 이미지를 생성할 intermediate vector 를 추출할 레이어의 이름
#                                       ('mapping_split1', 'mapping_split2' or 'w')
# - property_names        (list(str)) : 학습할 property name 의 리스트 (None 이면 'eyes', 'mouth', 'pose' 모두 학습)

# Returns:
# - mse_errors (dict) : 딥러닝 Neural Network 의 MSE Error 정보
#                       {'eyes': float, 'mouth': float, 'pose': float}

def run_stylegan_vector_find_gradient(finetune_v9_generator, device, n, layer_name, property_names=None):
    property_score_cnn = load_merged_property_score_cnn(device)
    mse_errors = {}
    model_dir_path = f'{PROJECT_DIR_PATH}/stylegan/models'

    # intermediate vector 샘플링 (전체 이미지 대상 Gradient Neural Network 학습)
    sampling_start_at = time.time()
    mid_vectors_all, property_scores_dict = sample_vector_and_compute_property_scores(finetune_v9_generator,
                                                                                      property_score_cnn,
                                                                                      layer_name,
                                                                                      n)

    print(f'sampling (from latent vector {layer_name}) running time (s) : {time.time() - sampling_start_at}\n')

    if property_names is None:
        property_names = PROPERTY_NAMES

    # 각 Property (eyes, mouth, pose) 별, Deep Learning (간단한 Neural Network) 학습
    for property_name in property_names:
        vectorfind_v9_gradient_nn = define_nn_model(layer_name)

        if property_name == property_names[0]:
            create_gradient_nn_structure_pdf(vectorfind_v9_gradient_nn, layer_name)

        input_data = mid_vectors_all
        output_data = property_scores_dict[property_name]
        data_np = np.concatenate([input_data, output_data], axis=1)

        mid_vector_dim = get_mid_vector_dim(layer_name)
        dataset = DatasetForSimpleNNForVectorFindV9(data_np, mid_vector_dim)
        train_loader, valid_loader, test_loader = create_dataloader(dataset)

        _, best_epoch_model = run_train_process(vectorfind_v9_gradient_nn,
                                                SimpleNNForVectorFindV9,
                                                mid_vector_dim,
                                                train_loader,
                                                valid_loader)

        # 해당 Neural Network 저장
        model_path = f'{model_dir_path}/stylegan_gen_vector_find_v9_nn_{property_name}.pth'
        torch.save(best_epoch_model.state_dict(), model_path)

        # 해당 Neural Network 를 테스트
        best_epoch_model.to(device)
        best_epoch_model.device = device
        performance_scores = run_test_process(best_epoch_model, test_loader)

        print(f'test performance scores ({property_name}) : {performance_scores}\n\n')
        mse_errors[property_name] = performance_scores['mse']

    return mse_errors
