import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from torchinfo import summary

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np

import math
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from global_common.torch_training import run_train, run_validation
from common import resize_and_normalize_img, DiagramImageDataset, IMG_HEIGHT, IMG_WIDTH


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/training_data'
K_FOLDS = 5
EARLY_STOPPING_ROUNDS = 10

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4


class BaseScoreCNN(nn.Module):
    def __init__(self):
        super(BaseScoreCNN, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x[:, :, IMG_HEIGHT // 4 : 3 * IMG_HEIGHT // 4, IMG_WIDTH // 4 : 3 * IMG_WIDTH // 4]

        # Conv
        x = self.conv1(x)  # 62
        x = self.conv2(x)  # 60
        x = self.pool1(x)  # 30

        x = self.conv3(x)  # 28
        x = self.pool2(x)  # 14

        x = self.conv4(x)  # 12
        x = self.pool3(x)  # 6

        x = self.conv5(x)  # 4

        x = x.view(-1, 256 * 4 * 4)

        # Fully Connected
        x = self.fc1(x)
        x = self.fc_final(x)

        return x


# 데이터셋 로딩
# Create Date : 2025.03.23
# Last Update Date : 2025.03.24
# - TRAIN_BATCH_SIZE, TEST_BATCH_SIZE 의 pre-defined value 이용
# - test data loader 에 test image path 의 list 를 별도 추가

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터 정보가 저장된 Pandas DataFrame
#                                   columns = ['img_path', 'score']

# Returns:
# - train_loader (DataLoader) : Train 데이터셋을 로딩한 PyTorch DataLoader
# - test_loader  (DataLoader) : Test 데이터셋을 로딩한 PyTorch DataLoader

def load_dataset(dataset_df):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    diagram_image_dataset = DiagramImageDataset(dataset_df, transform=transform)

    dataset_size = len(diagram_image_dataset)
    train_size = int(0.8 * dataset_size)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, dataset_size))

    train_dataset = Subset(diagram_image_dataset, train_indices)
    test_dataset = Subset(diagram_image_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    test_indices_original = list(filter(lambda x: x % 8 == 0, test_indices))
    test_indices_original_idxs = [idx // 8 for idx in test_indices_original]
    test_loader.test_img_paths = [diagram_image_dataset.img_paths[i] for i in test_indices_original_idxs]

    print(f'size of train loader : {len(train_loader.dataset)}')
    print(f'size of train loader : {len(test_loader.dataset)}')

    # test code
    tvt_labels = ['train', 'test']
    loaders = [train_loader, test_loader]
    test_dir_path = f'{PROJECT_DIR_PATH}/final_recommend_score/temp_test'

    for tvt_label, loader in zip(tvt_labels, loaders):
        for i in range(40):
            img = loader.dataset.__getitem__(i)[0]
            os.makedirs(test_dir_path, exist_ok=True)
            save_image(img, f'{test_dir_path}/{tvt_label}_data_{i:02d}.png')

    return train_loader, test_loader


# 모델 학습 실시 (K-Fold Cross Validation)
# Create Date : 2025.03.23
# Last Update Date : 2025.03.24
# - try-until-success 시스템 적용 (학습 실패 시 성공할 때까지 재학습)
# - TRAIN_BATCH_SIZE, VALID_BATCH_SIZE 의 pre-defined value 이용

# Arguments:
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader

# Returns:
# - cnn_models (list(nn.Module)) : 학습된 CNN Model 의 리스트 (총 K 개의 모델)
# - log/cnn_train_log.csv 파일에 학습 로그 저장

def train_cnn(data_loader):

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training : {device}')

    if 'cuda' in str(device):
        print(f'cuda is available with device {torch.cuda.get_device_name()}')

    # create models
    cnn_models = []

    for i in range(K_FOLDS):
        cnn_model = define_cnn_model()
        cnn_models.append(cnn_model)

    summary(cnn_models[0], input_size=(TRAIN_BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH))

    # Split train data using K-fold (5 folds)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)
    kfold_splitted_dataset = kfold.split(data_loader.dataset)

    # for train log
    os.makedirs(f'{PROJECT_DIR_PATH}/final_recommend_score/log', exist_ok=True)
    train_log = {'fold_no': [], 'success': [], 'min_val_loss': [],
                 'total_epochs': [], 'best_epoch': [],
                 'elapsed_time (s)': [], 'val_loss_list': []}

    # run training and validation
    for fold, (train_idxs, valid_idxs) in enumerate(kfold_splitted_dataset):
        print(f'=== training model {fold + 1} / {K_FOLDS} ===')

        # 학습 성공 (valid BCE loss 최솟값 <= 0.37) 시까지 반복
        while True:
            model = cnn_models[fold].to(device)
            model.device = device

            # train model
            start_at = time.time()
            val_loss_list, best_epoch_model = train_cnn_each_model(model, data_loader, train_idxs, valid_idxs)
            train_time = round(time.time() - start_at, 2)

            # check validation loss and success/fail of training
            min_val_loss = min(val_loss_list)
            min_val_loss_ = round(min_val_loss, 4)
            total_epochs = len(val_loss_list)
            val_loss_list_ = list(map(lambda x: round(x, 4), val_loss_list))

            is_train_successful = (min_val_loss <= 0.37)

            # test best epoch model correctly returned
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idxs)
            valid_loader = torch.utils.data.DataLoader(data_loader.dataset,
                                                       batch_size=VALID_BATCH_SIZE,
                                                       sampler=valid_sampler)

            _, val_loss_check = run_validation(model=best_epoch_model,
                                               valid_loader=valid_loader,
                                               device=model.device,
                                               loss_func=nn.BCELoss(reduction='sum'))

            print(f'best epoch model val_loss : {val_loss_check:.6f}, reported min val_loss : {min_val_loss:.6f}')
            assert abs(val_loss_check - min_val_loss) < 2e-5, "BEST MODEL VALID LOSS CHECK FAILED"

            # update train log
            train_log['fold_no'].append(fold)
            train_log['success'].append(is_train_successful)
            train_log['min_val_loss'].append(min_val_loss_)
            train_log['total_epochs'].append(total_epochs)
            train_log['best_epoch'].append(np.argmin(val_loss_list_))
            train_log['elapsed_time (s)'].append(train_time)
            train_log['val_loss_list'].append(val_loss_list_)

            train_log_path = f'{PROJECT_DIR_PATH}/final_recommend_score/log/cnn_train_log.csv'
            pd.DataFrame(train_log).to_csv(train_log_path)

            # train successful -> save model and then finish
            if is_train_successful:
                print('train successful!')

                model_path = f'{PROJECT_DIR_PATH}/final_recommend_score/models'
                os.makedirs(model_path, exist_ok=True)

                model_save_path = f'{model_path}/model_{fold}.pt'
                torch.save(best_epoch_model.state_dict(), model_save_path)
                break

            # train failed -> train new model
            else:
                print('train failed, retry ...')
                cnn_models[fold] = define_cnn_model()

    return cnn_models


# CNN 모델 정의
# Create Date : 2025.03.24
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - cnn_model (nn.Module) : 정의된 CNN 모델

def define_cnn_model():
    cnn_model = BaseScoreCNN()
    cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.001)
    cnn_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=cnn_model.optimizer,
                                                                     T_max=10,
                                                                     eta_min=0)

    return cnn_model


# 각 모델 (각각의 Fold 에 대한) 의 학습 실시
# Create Date : 2025.03.24
# Last Update Date : -

# Arguments:
# - model       (nn.Module)  : 학습 대상 CNN 모델
# - data_loader (DataLoader) : 데이터셋을 로딩한 PyTorch DataLoader
# - train_idxs  (np.array)   : 학습 데이터로 지정된 인덱스
# - valid_idxs  (np.array)   : 검증 데이터로 지정된 인덱스

# Returns:
# - val_loss_list    (list(float)) : valid loss 기록
# - best_epoch_model (nn.Module)   : 가장 낮은 valid loss 에서의 CNN 모델

def train_cnn_each_model(model, data_loader, train_idxs, valid_idxs):
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idxs)

    train_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=VALID_BATCH_SIZE, sampler=valid_sampler)

    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model = None

    val_loss_list = []

    # loss function
    loss_func = nn.BCELoss(reduction='sum')

    while True:
        train_loss = run_train(model=model,
                               train_loader=train_loader,
                               device=model.device,
                               loss_func=loss_func)

        _, val_loss = run_validation(model=model,
                                     valid_loader=valid_loader,
                                     device=model.device,
                                     loss_func=loss_func)

        print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}, val_loss : {val_loss:.4f}')

        val_loss_cpu = float(val_loss.detach().cpu())
        val_loss_list.append(val_loss_cpu)

        model.scheduler.step()

        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = current_epoch

            best_epoch_model = BaseScoreCNN().to(model.device)
            best_epoch_model.load_state_dict(model.state_dict())

        if current_epoch - min_val_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return val_loss_list, best_epoch_model


# 모델 불러오기
# Create Date : 2025.03.24
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - cnn_models (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)

def load_cnn_model():
    cnn_models = []

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for loading model : {device}')

    # add CNN models
    for i in range(K_FOLDS):
        model = BaseScoreCNN()
        model_path = f'{PROJECT_DIR_PATH}/final_recommend_score/models/model_{i}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        cnn_models.append(model)

    return cnn_models


# 학습된 모델을 이용하여 주어진 이미지에 대해 기본 가독성 점수 예측 (Ensemble 의 아이디어 / K 개 모델의 평균으로)
# Create Date : 2025.03.24
# Last Update Date : -

# Arguments:
# - test_loader (DataLoader)      : 테스트 데이터셋을 로딩한 PyTorch DataLoader
# - cnn_models  (list(nn.Module)) : load 된 CNN Model 의 리스트 (총 K 개의 모델)
# - report_path (str)             : final_score 의 report 를 저장할 경로

# Returns:
# - final_score        (Pandas DataFrame) : 최종 예측 점수 및 각 모델별 예측 점수 (0.0 ~ 5.0) 를 저장한 Pandas DataFrame
#                                           columns = ['img_path', 'rotate_angle', 'flip', 'true_score',
#                                                      'mean_pred_score', 'model0_score', 'model1_score', ...]
# - performance_scores (dict)             : 모델 성능 평가 결과 (0 ~ 5 점 척도 기준의 MAE, MSE, RMSE Error)
#                                           {'mae': float, 'mse': float, 'rmse': float}

def predict_score_image(test_loader, cnn_models, report_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for testing model : {device}')

    final_score_dict = {'img_path': [], 'rotate_angle': [], 'flip': [], 'true_score': [], 'mean_pred_score': []}

    for i in range(K_FOLDS):
        final_score_dict[f'model{i}_score'] = []
        cnn_models[i] = cnn_models[i].to(device)

    # do prediction and generate prediction result & score report
    for idx, (images, labels) in enumerate(test_loader):
        if idx % 100 == 0:
            print(f'batch {idx} of test dataset')

        with torch.no_grad():
            images, labels = images.to(device), labels.to(device).to(torch.float32)
            labels_cpu = labels.detach().cpu()
            current_batch_size = labels.size(0)

            # add image info
            for i in range(current_batch_size):
                data_loader_idx = idx * TEST_BATCH_SIZE + i
                img_idx = data_loader_idx // (4 * 2)

                img_path = test_loader.test_img_paths[img_idx]
                rotate_angle = (data_loader_idx % 4) * 90
                flip = 'vertical' if (data_loader_idx // 4) % 2 == 1 else 'none'

                final_score_dict['img_path'].append(img_path)
                final_score_dict['rotate_angle'].append(rotate_angle)
                final_score_dict['flip'].append(flip)

                rounded_true_score = round(5.0 * float(labels_cpu[i]), 2)
                final_score_dict['true_score'].append(rounded_true_score)

            # add model prediction scores
            model_scores = np.zeros((current_batch_size, K_FOLDS))

            for model_idx, model in enumerate(cnn_models):
                outputs_cpu = model(images).to(torch.float32).detach().cpu()

                for i in range(current_batch_size):
                    model_scores[i][model_idx] = 5.0 * outputs_cpu[i]
                    final_score_dict[f'model{model_idx}_score'].append(5.0 * float(outputs_cpu[i]))

            for i in range(current_batch_size):
                final_score_dict[f'mean_pred_score'].append(np.mean(model_scores[i]))

    # create final report
    true_scores = final_score_dict['true_score']
    pred_scores = final_score_dict['mean_pred_score']

    mse = mean_squared_error(true_scores, pred_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = math.sqrt(mse)

    final_score = pd.DataFrame(final_score_dict)
    performance_scores = {'mse': mse, 'mae': mae, 'rmse': rmse}

    final_score.to_csv(report_path, index=False)

    return final_score, performance_scores


if __name__ == '__main__':

    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/final_recommend_score/scores.csv')
    log_dir = f'{PROJECT_DIR_PATH}/final_recommend_score/log'

    # resize images to (128, 128)
    img_paths = dataset_df['img_path'].tolist()
    resize_and_normalize_img(img_paths,
                             train_data_dir_path=TRAIN_DATA_DIR_PATH,
                             dest_width=IMG_WIDTH,
                             dest_height=IMG_HEIGHT)

    # load dataset
    dataset_df = dataset_df.sample(frac=1, random_state=20250324)  # shuffle image sample order
    train_loader, test_loader = load_dataset(dataset_df)

    # load or train model
    try:
        print('loading CNN models ...')
        cnn_models = load_cnn_model()
        print('loading CNN models successful!')

    except Exception as e:
        print(f'CNN model load failed : {e}')
        cnn_models = train_cnn(train_loader)

    # performance evaluation
    report_path = f'{log_dir}/cnn_test_result.csv'
    _, performance_scores = predict_score_image(test_loader, cnn_models, report_path)

    print('PERFORMANCE SCORE :\n')
    print(performance_scores)

