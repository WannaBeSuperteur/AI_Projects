
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchinfo import summary

from sklearn.model_selection import KFold, StratifiedKFold

import os
import sys
import time

import pandas as pd
import numpy as np

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
TOP_DIR_PATH = os.path.dirname(os.path.abspath(PROJECT_DIR_PATH))
IMAGE_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results'

sys.path.append(TOP_DIR_PATH)
from global_common.torch_training import run_train, run_validation_detail

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
INFERENCE_BATCH_SIZE = 4

EARLY_STOPPING_ROUNDS = 10
K_FOLDS = 5

IMG_HEIGHT = 256
IMG_WIDTH = 256

base_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])


# CNN 으로 학습할 2,000 장에 대한 Dataset
class CNNImageDataset(Dataset):
    def __init__(self, dataset_df, transform, property_name):
        self.img_paths = dataset_df['img_path'].tolist()
        self.img_nos = dataset_df['img_no'].tolist()
        self.gender_scores = dataset_df['gender'].tolist()
        self.quality_scores = dataset_df['quality'].tolist()

        self.transform = transform
        self.property_name = property_name

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = f'{IMAGE_DATA_DIR_PATH}/{self.img_nos[idx]:06d}.jpg'
        image = read_image(img_path)

        # resize and normalize
        image = self.transform(image)

        if self.property_name == 'gender':
            score = self.gender_scores[idx]
        elif self.property_name == 'quality':
            score = self.quality_scores[idx]
        else:
            raise Exception("property_name must be one of ['gender', 'quality'].")

        return image, score


# 나머지 8,000 장에 대한 Dataset
class RemainingImageDataset(Dataset):
    def __init__(self, dataset_df, transform, property_name):
        self.img_paths = dataset_df['img_path'].tolist()
        self.img_nos = dataset_df['img_no'].tolist()

        self.transform = transform
        self.property_name = property_name

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = f'{IMAGE_DATA_DIR_PATH}/{self.img_nos[idx]:06d}.jpg'
        image = read_image(img_path)

        # resize and normalize
        image = self.transform(image)
        return image


# Original StyleGAN 이 생성한 이미지 중 첫 2,000 장의 데이터셋 생성을 위한 정보가 있는 Pandas DataFrame 생성
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - labeled_df (Pandas DataFrame) : 2,000 장의 데이터를 train data 로 하는 데이터셋 생성을 위한 Pandas DataFrame
#                                   columns = ['img_path', 'img_no', 'gender', 'quality']

def create_train_dataset_df():
    labeled_data_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/synthesize_results_quality_and_gender.csv'
    labeled_df = pd.read_csv(labeled_data_path)

    labeled_df['img_no'] = labeled_df['img_name']
    labeled_df['img_path'] = labeled_df['img_name'].apply(lambda x: f'{IMAGE_DATA_DIR_PATH}/{x:06d}.jpg')

    labeled_df.drop(columns=['img_name'], inplace=True)

    return labeled_df


# Original StyleGAN 이 생성한 이미지 중 첫 2,000 장의 데이터셋 로딩
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name (str) : 핵심 속성 값 이름 ('gender' or 'quality')

# Returns:
# - data_loader (DataLoader) : 2,000 장의 데이터를 train data 로 하는 DataLoader

def load_dataset(property_name):
    dataset_df = create_train_dataset_df()
    dataset = CNNImageDataset(dataset_df, transform=base_transform, property_name=property_name)

    data_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    return data_loader


# Original StyleGAN 이 생성한 이미지 중 나머지 8,000 장 로딩
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name (str) : 핵심 속성 값 이름 ('gender' or 'quality')

# Returns:
# - data_loader (DataLoader) : 나머지 8,000 장의 데이터를 test (inference) data 로 하는 DataLoader

def load_remaining_images_dataset(property_name):
    img_names = os.listdir(IMAGE_DATA_DIR_PATH)
    img_paths = [f'{IMAGE_DATA_DIR_PATH}/{name}' for name in img_names]

    remaining_dataset_dict = {'img_path': img_paths, 'img_no': img_names}
    remaining_dataset_df = pd.DataFrame(remaining_dataset_dict)

    remaining_dataset = RemainingImageDataset(remaining_dataset_df,
                                              transform=base_transform,
                                              property_name=property_name)

    remaining_data_loader = DataLoader(remaining_dataset,
                                       batch_size=INFERENCE_BATCH_SIZE,
                                       shuffle=False)

    return remaining_data_loader


# CNN 모델 정의
# Create Date : 2025.04.10
# Last Update Date : 2025.04.10
# -> learning rate 1e-3 (학습 실패) -> 5e-5 (기존의 5% 로 초기값 감소)

# Arguments:
# - cnn_model_class (nn.Module class) : 학습할 CNN 모델의 Class
# - device          (device)          : CNN 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - cnn_model (nn.Module) : 학습할 CNN 모델

def define_cnn_model(cnn_model_class, device):
    cnn_model = cnn_model_class()
    cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.00005)
    cnn_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=cnn_model.optimizer,
                                                                     T_max=10,
                                                                     eta_min=0)

    cnn_model.to(device)
    cnn_model.device = device

    return cnn_model


# 모델 학습 실시 (K-Fold / Stratified K-Fold Cross Validation)
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - data_loader     (DataLoader)      : 2,000 장의 데이터를 train data 로 하는 DataLoader
# - is_stratified   (bool)            : True 이면 Stratified K-Fold, False 이면 일반 K-Fold Cross Validation 적용
# - property_name   (str)             : 핵심 속성 값 이름 ('gender' or 'quality')
# - cnn_model_class (nn.Module class) : 학습할 CNN 모델의 Class

# Returns:
# - cnn_models (list(nn.Module)) : 학습된 CNN Model 의 리스트 (총 K 개의 모델)
# - log/cnn_train_log.csv 파일에 학습 로그 저장

def train_cnn_models(data_loader, is_stratified, property_name, cnn_model_class):

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training : {device}')

    if 'cuda' in str(device):
        print(f'cuda is available with device {torch.cuda.get_device_name()}')

    # create models
    cnn_models = []

    for i in range(K_FOLDS):
        cnn_model = define_cnn_model(cnn_model_class=cnn_model_class, device=device)
        cnn_models.append(cnn_model)

    summary(cnn_models[0], input_size=(TRAIN_BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH))

    # Split train data using K-fold or Stratified K-fold (5 folds)
    if is_stratified:
        kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True)
    else:
        kfold = KFold(n_splits=K_FOLDS, shuffle=True)

    kfold_splitted_dataset = kfold.split(data_loader.dataset)

    # for train log
    os.makedirs(f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/log', exist_ok=True)
    train_log = {'fold_no': [], 'success': [], 'min_val_loss': [],
                 'total_epochs': [], 'best_epoch': [],
                 'elapsed_time (s)': [], 'val_loss_list': []}

    # threshold for train success check & detailed performance report per epoch
    if property_name == 'gender':
        val_loss_threshold, pos_neg_threshold = 0.25, 0.5
    elif property_name == 'quality':
        val_loss_threshold, pos_neg_threshold = 0.11, 0.9
    else:
        raise Exception("property_name must be one of ['gender', 'quality'].")

    # run training and validation
    for fold, (train_idxs, valid_idxs) in enumerate(kfold_splitted_dataset):
        print(f'=== training model {fold + 1} / {K_FOLDS} ===')

        # 학습 성공 (valid BCE loss 최솟값 <= threshold) 시까지 반복
        while True:
            model = cnn_models[fold].to(device)
            model.device = device

            # train model
            start_at = time.time()
            val_loss_list, best_epoch_model, performance_each_epoch = train_cnn_each_model(model,
                                                                                           data_loader,
                                                                                           train_idxs,
                                                                                           valid_idxs,
                                                                                           cnn_model_class,
                                                                                           pos_neg_threshold)
            train_time = round(time.time() - start_at, 2)

            # check train successful and save train log
            best_epoch_model.device = device

            save_train_log(val_loss_list, val_loss_threshold, best_epoch_model, data_loader, valid_idxs,
                           property_name, train_log, fold, train_time)

            is_train_successful = min(val_loss_list) <= val_loss_threshold

            # train successful -> save model and then finish
            if is_train_successful:
                print('train successful!')

                model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/models'
                os.makedirs(model_path, exist_ok=True)

                model_save_path = f'{model_path}/{property_name}_model_{fold}.pt'
                torch.save(best_epoch_model.state_dict(), model_save_path)

                performance_log_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/log'
                performance_log_save_path = f'{performance_log_path}/epoch_detail_{property_name}_{fold}.csv'
                performance_each_epoch.to_csv(performance_log_save_path, index=False)

                break

            # train failed -> train new model
            else:
                print('train failed, retry ...')
                cnn_models[fold] = define_cnn_model(cnn_model_class, device)

    raise NotImplementedError


# 학습 로그 저장
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - val_loss_list      (list(float)) : valid loss 기록
# - val_loss_threshold (float)       : valid loss 의 threshold (해당 값 이하 => 학습 성공으로 간주)
# - best_epoch_model   (nn.Module)   : 가장 낮은 valid loss 에서의 CNN 모델
# - data_loader        (DataLoader)  : 2,000 장의 데이터를 train data 로 하는 DataLoader
# - valid_idxs         (np.array)    : 검증 데이터로 지정된 인덱스
# - property_name      (str)         : 핵심 속성 값 이름 ('gender' or 'quality')
# - train_log          (dict)        : 학습 로그 (keys: ['fold_no', 'success', 'min_val_loss', 'total_epochs',
#                                                       'best_epoch', 'elapsed_time (s)', val_loss_list']
# - fold               (int)         : fold 번호
# - train_time         (float)       : 학습 소요 시간

# Returns:
# - 없음

def save_train_log(val_loss_list, val_loss_threshold, best_epoch_model, data_loader, valid_idxs, property_name,
                   train_log, fold, train_time):

    # check validation loss and success/fail of training
    min_val_loss = min(val_loss_list)
    min_val_loss_ = round(min_val_loss, 4)
    total_epochs = len(val_loss_list)
    val_loss_list_ = list(map(lambda x: round(x, 4), val_loss_list))

    # test best epoch model correctly returned
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idxs)
    valid_loader = torch.utils.data.DataLoader(data_loader.dataset,
                                               batch_size=VALID_BATCH_SIZE,
                                               sampler=valid_sampler)

    val_result_check = run_validation_detail(model=best_epoch_model,
                                             valid_loader=valid_loader,
                                             device=best_epoch_model.device,
                                             loss_func=nn.BCELoss(reduction='sum'))
    val_loss_check = val_result_check['val_loss']

    print(f'best epoch model val_loss : {val_loss_check:.6f}, reported min val_loss : {min_val_loss:.6f}')
    assert abs(val_loss_check - min_val_loss) < 2e-5, "BEST MODEL VALID LOSS CHECK FAILED"

    # update train log
    train_log['fold_no'].append(fold)
    train_log['success'].append(min_val_loss < val_loss_threshold)
    train_log['min_val_loss'].append(min_val_loss_)
    train_log['total_epochs'].append(total_epochs)
    train_log['best_epoch'].append(np.argmin(val_loss_list_))
    train_log['elapsed_time (s)'].append(train_time)
    train_log['val_loss_list'].append(val_loss_list_)

    train_log_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/log/train_log_{property_name}.csv'
    pd.DataFrame(train_log).to_csv(train_log_path)


# 각 모델 (각각의 Fold 에 대한) 의 학습 실시
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - model             (nn.Module)       : 학습 대상 CNN 모델
# - data_loader       (DataLoader)      : 2,000 장의 데이터를 train data 로 하는 DataLoader
# - train_idxs        (np.array)        : 학습 데이터로 지정된 인덱스
# - valid_idxs        (np.array)        : 검증 데이터로 지정된 인덱스
# - cnn_model_class   (nn.Module class) : 학습할 CNN 모델의 Class
# - pos_neg_threshold (float)           : Output Score (0 ~ 1) 에 대해 Positive / Negative 를 구분하기 위한 Threshold

# Returns:
# - val_loss_list          (list(float))      : valid loss 기록
# - performance_each_epoch (Pandas DataFrame) : 각 Epoch 에서의 상세한 성능 로그 (Accuracy, F1 Score 등 / threshold = 0.5 기준)
# - best_epoch_model       (nn.Module)        : 가장 낮은 valid loss 에서의 CNN 모델

def train_cnn_each_model(model, data_loader, train_idxs, valid_idxs, cnn_model_class, pos_neg_threshold):
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idxs)

    train_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=VALID_BATCH_SIZE, sampler=valid_sampler)

    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model = None

    val_loss_list = []
    performance_each_epoch_dict = {'epoch': [], 'val_loss': [], 'val_accuracy': [],
                                   'tp': [], 'tn': [], 'fp': [], 'fn': [],
                                   'val_recall': [], 'val_precision': [], 'val_f1_score': []}

    # loss function
    loss_func = nn.BCELoss(reduction='sum')

    while True:

        # run train and validation
        train_loss = run_train(model=model,
                               train_loader=train_loader,
                               device=model.device,
                               loss_func=loss_func)

        val_result = run_validation_detail(model=model,
                                           valid_loader=valid_loader,
                                           device=model.device,
                                           loss_func=loss_func,
                                           threshold=pos_neg_threshold)

        val_loss = val_result['val_loss']

        # add performance log
        performance_each_epoch_dict['epoch'].append(current_epoch)

        for metric_name, metric_value in val_result.items():
            if metric_name == 'val_loss':
                performance_each_epoch_dict[metric_name].append(float(val_loss.detach().cpu()))
            else:
                performance_each_epoch_dict[metric_name].append(metric_value)

        print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}, val_loss : {val_loss:.4f}')

        # add valid loss log
        val_loss_cpu = float(val_loss.detach().cpu())
        val_loss_list.append(val_loss_cpu)

        # update scheduler
        model.scheduler.step()

        # update best epoch
        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = current_epoch

            best_epoch_model = cnn_model_class().to(model.device)
            best_epoch_model.load_state_dict(model.state_dict())

        if current_epoch - min_val_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    performance_each_epoch = pd.DataFrame(performance_each_epoch_dict)
    return val_loss_list, performance_each_epoch, best_epoch_model


# 학습된 CNN 모델 불러오기
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name   (str)             : 핵심 속성 값 이름 ('gender' or 'quality')
# - cnn_model_class (nn.Module class) : 학습할 CNN 모델의 Class

# Returns:
# - cnn_models (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

def load_cnn_model(property_name, cnn_model_class):
    cnn_models = []

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for loading model : {device}')

    # add CNN models
    for i in range(K_FOLDS):
        model = cnn_model_class()
        model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentations/cnn/models/{property_name}_model_{i}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        model.to(device)
        model.device = device

        cnn_models.append(model)

    return cnn_models


# 학습된 모델을 이용하여 나머지 8,000 장의 이미지에 대해 핵심 속성 값 예측 (Ensemble 의 아이디어 / K 개 모델의 평균으로)
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - property_name           (str)             : 핵심 속성 값 이름 ('gender' or 'quality')
# - remaining_images_loader (DataLoader)      : 나머지 8,000 장의 이미지 데이터셋을 로딩한 PyTorch DataLoader
# - cnn_models              (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)
# - report_path             (str)             : final_score 의 report 를 저장할 경로

# Returns:
# - final_score (Pandas DataFrame) : 해당 속성 값에 대한 모델 예측값을 저장한 Pandas DataFrame
#                                    columns = ['img_no', 'img_path', 'property_{property_name}_final_score',
#                                               'score_model_0', 'score_model_1', ...]

def predict_score_remaining_images(property_name, remaining_images_loader, cnn_models, report_path):
    raise NotImplementedError
