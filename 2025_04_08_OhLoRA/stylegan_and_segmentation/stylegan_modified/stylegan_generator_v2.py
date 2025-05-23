import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np

from stylegan_modified.fine_tuning import concatenate_property_scores
from stylegan_modified.stylegan_generator_v2_compare_property import write_property_compare_result
from stylegan_modified.stylegan_generator_v2_gen_model import train_stylegan_finetune_v2
from stylegan_modified.stylegan_generator_v2_cnn import PropertyScoreCNN
from stylegan_modified.stylegan_generator import StyleGANGenerator

from torch.utils.data import random_split, DataLoader
from torchinfo import summary

torch.set_printoptions(linewidth=160, sci_mode=False)
np.set_printoptions(suppress=True)


IMG_HEIGHT = 256
IMG_WIDTH = 256
PROPERTY_DIMS_Z = 7  # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
EARLY_STOPPING_ROUNDS = 20

VALID_OUTPUT_LABEL_LOG_CNT_PER_EPOCH = 30

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
INFERENCE_RESULT_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/inference_result'

cnn_loss_func = nn.MSELoss(reduction='mean')
cnn_valid_log = {'epoch': [], 'img_path': [],
                 'eyes_score_output': [], 'eyes_score_label': [],
                 'hair_color_score_output': [], 'hair_color_score_label': [],
                 'hair_length_score_output': [], 'hair_length_score_label': [],
                 'mouth_score_output': [], 'mouth_score_label': [],
                 'pose_score_output': [], 'pose_score_label': [],
                 'back_mean_score_output': [], 'back_mean_score_label': [],
                 'back_std_score_output': [], 'back_std_score_label': []}


# CNN 모델 정의
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - device (device) : CNN 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - cnn_model (nn.Module) : 학습할 CNN 모델

def define_cnn_model(device):
    cnn_model = PropertyScoreCNN()
    cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.00005)
    cnn_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=cnn_model.optimizer,
                                                                     T_max=10,
                                                                     eta_min=0)

    cnn_model.to(device)
    cnn_model.device = device

    # print summary of CNN model before training
    summary(cnn_model, input_size=(TRAIN_BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH))

    return cnn_model


# CNN 모델 학습 (Property Score 계산용)
# Create Date : 2025.04.13
# Last Update Date : 2025.04.14
# - train log 에 loss 외에도 abs diff, corr-coef 추가
# - Train Loss 계산을 제거하여 학습 속도 향상 시도

# Arguments:
# - device                 (device)     : CNN 모델을 mapping 시킬 device (GPU 등)
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - best_epoch_model (nn.Module) : 학습된 CNN 모델 (valid loss 가 가장 작은 best epoch)

def train_cnn_model(device, fine_tuning_dataloader):
    cnn_model = define_cnn_model(device)

    # split dataset
    dataset_size = len(fine_tuning_dataloader.dataset)
    train_size = int(dataset_size * 0.9)
    valid_size = dataset_size - train_size

    train_dataset, valid_dataset = random_split(fine_tuning_dataloader.dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    # prepare training
    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model = None

    performance_dict = {'epoch': [], 'valid_loss': [], 'val_loss_except_back_std': [],
                        'eyes_score_loss': [], 'hair_color_score_loss': [], 'hair_length_score_loss': [],
                        'mouth_score_loss': [], 'pose_score_loss': [],
                        'back_mean_score_loss': [], 'back_std_score_loss': [],
                        'eyes_score_abs_diff': [], 'hair_color_score_abs_diff': [], 'hair_length_score_abs_diff': [],
                        'mouth_score_abs_diff': [], 'pose_score_abs_diff': [],
                        'back_mean_score_abs_diff': [], 'back_std_score_abs_diff': [],
                        'eyes_score_corr': [], 'hair_color_score_corr': [], 'hair_length_score_corr': [],
                        'mouth_score_corr': [], 'pose_score_corr': [],
                        'back_mean_score_corr': [], 'back_std_score_corr': []}

    # run training until early stopping
    while True:

        # run train and validation
        train_cnn_train_step(cnn_model=cnn_model,
                             cnn_train_dataloader=train_loader)

        valid_log = train_cnn_valid_step(cnn_model=cnn_model,
                                         cnn_valid_dataloader=valid_loader,
                                         current_epoch=current_epoch)

        val_loss = valid_log['valid_loss']
        val_loss_early_stop = (val_loss * PROPERTY_DIMS_Z - valid_log['back_std_score_loss']) / (PROPERTY_DIMS_Z - 1)
        print(f'epoch : {current_epoch}, valid_loss : {val_loss:.4f} (except back std : {val_loss_early_stop:.4f})')

        # update log
        for k, v in valid_log.items():
            if k == 'epoch':
                performance_dict[k].append(v)
            else:
                performance_dict[k].append(round(v, 4))

        performance_dict['val_loss_except_back_std'].append(round(val_loss_early_stop, 4))

        # save train log
        train_log_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/train_log_v2_cnn.csv'
        performance_df = pd.DataFrame(performance_dict)
        performance_df.to_csv(train_log_path, index=False)

        # update scheduler
        cnn_model.scheduler.step()

        # update best epoch
        if min_val_loss is None or val_loss_early_stop < min_val_loss:
            min_val_loss = val_loss_early_stop
            min_val_loss_epoch = current_epoch

            best_epoch_model = PropertyScoreCNN().to(cnn_model.device)
            best_epoch_model.load_state_dict(cnn_model.state_dict())

        if current_epoch - min_val_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return best_epoch_model


# CNN 모델의 Train Step
# Create Date : 2025.04.13
# Last Update Date : 2025.04.14
# - Train Loss 계산을 제거하여 학습 속도 향상 시도

# Arguments:
# - cnn_model            (nn.Module)  : 학습 중인 CNN 모델
# - cnn_train_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 CNN Train 용 Data Loader

def train_cnn_train_step(cnn_model, cnn_train_dataloader):
    cnn_model.train()

    for idx, raw_data in enumerate(cnn_train_dataloader):
        images = raw_data['image']
        labels = concatenate_property_scores(raw_data)
        images, labels = images.to(cnn_model.device), labels.to(cnn_model.device).to(torch.float32)

        # train 실시
        cnn_model.optimizer.zero_grad()
        outputs = cnn_model(images).to(torch.float32)

        loss = cnn_loss_func(outputs, labels)
        loss.backward()
        cnn_model.optimizer.step()

#        if idx % 5 == 0:
#            print(idx, 'train outputs:\n', outputs[:4])
#            print(idx, 'train labels:\n', labels[:4])


# CNN 모델의 Valid Step
# Create Date : 2025.04.13
# Last Update Date : 2025.04.14
# - train log 에 loss 외에도 abs diff, corr-coef 추가
# - Valid Output 및 Valid Label 저장 부분 추가 (with image paths)

# Arguments:
# - cnn_model            (nn.Module)  : 학습 중인 CNN 모델
# - cnn_valid_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 CNN Validation 용 Data Loader
# - current_epoch        (int)        : 현재 epoch 번호 (로깅 목적)

# Returns:
# - valid_log (dict) : CNN 모델의 Validation Log
#                      {'epoch': int, 'valid_loss': float,
#                       'eyes_score_loss': float, ..., 'back_std_score_loss': float,
#                       'eyes_score_abs_diff': float, ..., 'back_std_score_abs_diff': float,
#                       'eyes_score_corr': float, ..., 'back_std_score_corr': float}

def train_cnn_valid_step(cnn_model, cnn_valid_dataloader, current_epoch):
    cnn_model.eval()
    correct, total = 0, 0
    val_loss_sum = 0

    valid_log = {'epoch': current_epoch,
                 'eyes_score_loss': 0.0, 'hair_color_score_loss': 0.0, 'hair_length_score_loss': 0.0,
                 'mouth_score_loss': 0.0, 'pose_score_loss': 0.0,
                 'back_mean_score_loss': 0.0, 'back_std_score_loss': 0.0,
                 'eyes_score_abs_diff': 0.0, 'hair_color_score_abs_diff': 0.0, 'hair_length_score_abs_diff': 0.0,
                 'mouth_score_abs_diff': 0.0, 'pose_score_abs_diff': 0.0,
                 'back_mean_score_abs_diff': 0.0, 'back_std_score_abs_diff': 0.0}

    outputs_list = []
    labels_list = []
    img_path_list = []

    with torch.no_grad():
        for idx, raw_data in enumerate(cnn_valid_dataloader):
            images = raw_data['image']
            labels = concatenate_property_scores(raw_data)
            images, labels = images.to(cnn_model.device), labels.to(cnn_model.device).to(torch.float32)

            outputs = cnn_model(images).to(torch.float32)

            val_loss_current_batch = float(cnn_loss_func(outputs, labels).detach().cpu().numpy())
            val_loss_sum += val_loss_current_batch * labels.size(0)
            total += labels.size(0)

            # aggregate outputs and labels
            outputs_list += list(outputs.detach().cpu().numpy())
            labels_list += list(labels.detach().cpu().numpy())
            img_path_list += list(raw_data['img_path'])

            # compute detailed losses and abs. diff info
            compute_detailed_valid_results(outputs, labels, valid_log)

#            if idx % 5 == 0:
#                print(idx, 'valid outputs:\n', outputs)
#                print(idx, 'valid labels:\n', labels)

        # Final Loss 계산
        val_loss = val_loss_sum / total
        valid_log['valid_loss'] = val_loss

        for metric_name in ['loss', 'abs_diff']:
            valid_log[f'eyes_score_{metric_name}'] /= total
            valid_log[f'hair_color_score_{metric_name}'] /= total
            valid_log[f'hair_length_score_{metric_name}'] /= total
            valid_log[f'mouth_score_{metric_name}'] /= total
            valid_log[f'pose_score_{metric_name}'] /= total
            valid_log[f'back_mean_score_{metric_name}'] /= total
            valid_log[f'back_std_score_{metric_name}'] /= total

    # compute corr-coef
    valid_log['eyes_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 0], np.array(labels_list)[:, 0])[0][1]
    valid_log['hair_color_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 1], np.array(labels_list)[:, 1])[0][1]
    valid_log['hair_length_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 2], np.array(labels_list)[:, 2])[0][1]
    valid_log['mouth_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 3], np.array(labels_list)[:, 3])[0][1]
    valid_log['pose_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 4], np.array(labels_list)[:, 4])[0][1]
    valid_log['back_mean_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 5], np.array(labels_list)[:, 5])[0][1]
    valid_log['back_std_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 6], np.array(labels_list)[:, 6])[0][1]

    # save output list and labels list (for first 30 samples in valid dataset)
    save_valid_output_and_labels(outputs_list, labels_list, img_path_list, current_epoch)

    return valid_log


# Valid Output 및 Label 의 리스트를 csv 형태로 저장 (stylegan_modified/train_log_v2_cnn_val_log.csv)
# Create Date : 2025.04.14
# Last Update Date : 2025.04.14
# - Valid Output 및 Label 기록에 해당 image path 추가

# Arguments:
# - outputs_list  (list) : Valid Output List
# - labels_list   (list) : Valid Label List
# - img_path_list (list) : Valid Sample 에 대한 Image Path Lost
# - current_epoch (int)  : 현재 epoch 번호 (로깅 목적)

def save_valid_output_and_labels(outputs_list, labels_list, img_path_list, current_epoch):
    global cnn_valid_log

    value_types = ['output', 'label']
    value_list = [np.round(np.array(outputs_list[:VALID_OUTPUT_LABEL_LOG_CNT_PER_EPOCH]), 4),
                  np.array(labels_list[:VALID_OUTPUT_LABEL_LOG_CNT_PER_EPOCH])]

    cnn_valid_log['epoch'] += [current_epoch] * VALID_OUTPUT_LABEL_LOG_CNT_PER_EPOCH
    cnn_valid_log['img_path'] += img_path_list[:VALID_OUTPUT_LABEL_LOG_CNT_PER_EPOCH]

    for value_type, value_list in zip(value_types, value_list):
        cnn_valid_log[f'eyes_score_{value_type}'] += list(value_list[:, 0])
        cnn_valid_log[f'hair_color_score_{value_type}'] += list(value_list[:, 1])
        cnn_valid_log[f'hair_length_score_{value_type}'] += list(value_list[:, 2])
        cnn_valid_log[f'mouth_score_{value_type}'] += list(value_list[:, 3])
        cnn_valid_log[f'pose_score_{value_type}'] += list(value_list[:, 4])
        cnn_valid_log[f'back_mean_score_{value_type}'] += list(value_list[:, 5])
        cnn_valid_log[f'back_std_score_{value_type}'] += list(value_list[:, 6])

    cnn_valid_log_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/train_log_v2_cnn_val_log.csv'
    pd.DataFrame(cnn_valid_log).to_csv(cnn_valid_log_path, index=False)


# CNN 모델의 Valid Step 에서 상세한 Valid Loss 결과 (Loss, Abs Diff) 저장
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - outputs   (PyTorch Tensor) : Valid dataset 에 대한 predicted output
# - labels    (PyTorch Tensor) : Valid dataset 에 대한 label
# - valid_log (dict)           : CNN 모델의 Validation Log
#                                {'epoch': int, 'valid_loss': float,
#                                 'eyes_score_loss': float, ..., 'back_std_score_loss': float,
#                                 'eyes_score_abs_diff': float, ..., 'back_std_score_abs_diff': float,
#                                 'eyes_score_corr': float, ..., 'back_std_score_corr': float}

def compute_detailed_valid_results(outputs, labels, valid_log):

    # compute loss
    eyes_score_loss = float(cnn_loss_func(outputs[:, :1], labels[:, :1]).detach().cpu().numpy())
    hair_color_score_loss = float(cnn_loss_func(outputs[:, 1:2], labels[:, 1:2]).detach().cpu().numpy())
    hair_length_score_loss = float(cnn_loss_func(outputs[:, 2:3], labels[:, 2:3]).detach().cpu().numpy())
    mouth_score_loss = float(cnn_loss_func(outputs[:, 3:4], labels[:, 3:4]).detach().cpu().numpy())
    pose_score_loss = float(cnn_loss_func(outputs[:, 4:5], labels[:, 4:5]).detach().cpu().numpy())
    back_mean_score_loss = float(cnn_loss_func(outputs[:, 5:6], labels[:, 5:6]).detach().cpu().numpy())
    back_std_score_loss = float(cnn_loss_func(outputs[:, 6:], labels[:, 6:]).detach().cpu().numpy())

    valid_log['eyes_score_loss'] += eyes_score_loss * labels.size(0)
    valid_log['hair_color_score_loss'] += hair_color_score_loss * labels.size(0)
    valid_log['hair_length_score_loss'] += hair_length_score_loss * labels.size(0)
    valid_log['mouth_score_loss'] += mouth_score_loss * labels.size(0)
    valid_log['pose_score_loss'] += pose_score_loss * labels.size(0)
    valid_log['back_mean_score_loss'] += back_mean_score_loss * labels.size(0)
    valid_log['back_std_score_loss'] += back_std_score_loss * labels.size(0)

    # compute abs diff
    eyes_score_abs_diff = float(nn.L1Loss()(outputs[:, :1], labels[:, :1]).detach().cpu().numpy())
    hair_color_score_abs_diff = float(nn.L1Loss()(outputs[:, 1:2], labels[:, 1:2]).detach().cpu().numpy())
    hair_length_score_abs_diff = float(nn.L1Loss()(outputs[:, 2:3], labels[:, 2:3]).detach().cpu().numpy())
    mouth_score_abs_diff = float(nn.L1Loss()(outputs[:, 3:4], labels[:, 3:4]).detach().cpu().numpy())
    pose_score_abs_diff = float(nn.L1Loss()(outputs[:, 4:5], labels[:, 4:5]).detach().cpu().numpy())
    back_mean_score_abs_diff = float(nn.L1Loss()(outputs[:, 5:6], labels[:, 5:6]).detach().cpu().numpy())
    back_std_score_abs_diff = float(nn.L1Loss()(outputs[:, 6:], labels[:, 6:]).detach().cpu().numpy())

    valid_log['eyes_score_abs_diff'] += eyes_score_abs_diff * labels.size(0)
    valid_log['hair_color_score_abs_diff'] += hair_color_score_abs_diff * labels.size(0)
    valid_log['hair_length_score_abs_diff'] += hair_length_score_abs_diff * labels.size(0)
    valid_log['mouth_score_abs_diff'] += mouth_score_abs_diff * labels.size(0)
    valid_log['pose_score_abs_diff'] += pose_score_abs_diff * labels.size(0)
    valid_log['back_mean_score_abs_diff'] += back_mean_score_abs_diff * labels.size(0)
    valid_log['back_std_score_abs_diff'] += back_std_score_abs_diff * labels.size(0)


# 학습된 CNN 모델 불러오기
# Create Date : 2025.04.13
# Last Update Date : 2025.04.14
# - 모델 로딩 시 weights_only (가중치만 로딩) 를 False 로 설정

# Arguments:
# - cnn_model_path (str)    : CNN 모델 저장 경로
# - device         (device) : CNN 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - cnn_model (nn.Module) : 학습된 CNN 모델

def load_cnn_model(cnn_model_path, device):
    cnn_model = PropertyScoreCNN()
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device, weights_only=False))

    cnn_model.to(device)
    cnn_model.device = device

    return cnn_model


# Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator 불러오기
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - v2_gen_path (str)    : StyleGAN-FineTune-v2 모델 저장 경로
# - device      (device) : StyleGAN-FineTune-v2 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator

def load_stylegan_finetune_v2_model(v2_gen_path, device):
    fine_tuned_generator = StyleGANGenerator(resolution=IMG_HEIGHT)  # IMG_HEIGHT = IMG_WIDTH = RESOLUTION
    fine_tuned_generator.load_state_dict(torch.load(v2_gen_path, map_location=device, weights_only=False))

    fine_tuned_generator.to(device)
    fine_tuned_generator.device = device

    return fine_tuned_generator


# StyleGAN-FineTune-v2 모델 Fine-Tuning 실시
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - generator              (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator
# - cnn_model            (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델을 학습하기 위한 CNN
# - exist_dict           (dict)      : 각 모델 (CNN, StyleGAN-FineTune-v2) 의 존재 여부 (= 신규 학습 미 실시 여부)

def run_fine_tuning(generator, fine_tuning_dataloader):
    cnn_save_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v2_cnn.pth'

    cnn_model_exist = False
    stylegan_finetune_v2_exist = False

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v2 : {device}')

    # load CNN model
    try:
        cnn_model = load_cnn_model(cnn_save_path, device)
        cnn_model_exist = True

    except Exception as e:
        print(f'cnn model load failed : {e}')

        # train CNN model
        cnn_model = train_cnn_model(device, fine_tuning_dataloader)
        torch.save(cnn_model.state_dict(), cnn_save_path)

    # Compare Property Values (Can we use CNN-derived property values for image generate model?)
#    write_property_compare_result(fine_tuning_dataloader, cnn_model)

    # load or newly train Fine-Tuned Generator (StyleGAN-FineTune-v2)
    v2_gen_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v2.pth'

    try:
        fine_tuned_generator = load_stylegan_finetune_v2_model(v2_gen_path, device)
        stylegan_finetune_v2_exist = True

    except Exception as e:
        print(f'StyleGAN-FineTune-v2 model load failed : {e}')

        # train StyleGAN-FineTune-v2 model
        fine_tuned_generator = train_stylegan_finetune_v2(device, generator, cnn_model)
        torch.save(fine_tuned_generator.state_dict(), v2_gen_path)

    exist_dict = {'cnn': cnn_model_exist, 'stylegan_finetune_v2': stylegan_finetune_v2_exist}
    print(f'model existance : {exist_dict}')

    return fine_tuned_generator, cnn_model, exist_dict
