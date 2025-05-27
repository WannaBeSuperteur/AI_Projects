import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader, Dataset
from torchinfo import summary
from torchvision.io import read_image

import pandas as pd
import numpy as np
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMG_RESOLUTION = 256

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
EARLY_STOPPING_ROUNDS = 1
VALID_OUTPUT_LABEL_LOG_CNT_PER_EPOCH = 30

cnn_loss_func = nn.MSELoss(reduction='mean')
cnn_valid_log = {'epoch': [], 'img_path': [], 'hairstyle_score_output': [], 'hairstyle_score_label': []}

train_log_dir_path = f'{PROJECT_DIR_PATH}/property_score_cnn/train_logs'
os.makedirs(train_log_dir_path, exist_ok=True)

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# Image Dataset with Property Scores
class PropertyScoreImageDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_paths = dataset_df['img_path'].tolist()
        self.transform = transform

        self.hairstyle_scores = dataset_df['hairstyle_score'].tolist()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)

        hairstyle_score = self.hairstyle_scores[idx]
        property_scores = {'hairstyle': hairstyle_score}

        # normalize
        image = self.transform(image)

        # return simplified image path
        simplified_img_path = '/'.join(img_path.split('/')[-2:])

        return {'image': image, 'label': property_scores, 'img_path': simplified_img_path}


# Hairstyle Score CNN (곱슬머리 vs. 직모)

class HairstyleScorePartCNN(nn.Module):
    def __init__(self):
        super(HairstyleScorePartCNN, self).__init__()

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
            nn.Conv2d(64, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05)
        )

    def forward(self, x):

        # Conv
        x = self.conv1(x)  # 94 x 126
        x = self.conv2(x)  # 92 x 124
        x = self.pool1(x)  # 46 x 62

        x = self.conv3(x)  # 46 x 60
        x = self.pool2(x)  # 22 x 30

        x = self.conv4(x)  # 20 x 28
        x = self.pool3(x)  # 10 x 14

        x = self.conv5(x)  # 8 x 12
        x = self.conv6(x)  # 6 x 10

        x = x.view(-1, 128 * 6 * 10)
        return x


class HairstyleScoreCNN(nn.Module):
    def __init__(self):
        super(HairstyleScoreCNN, self).__init__()

        # Conv Layers
        self.bottom_left_cnn = HairstyleScorePartCNN()
        self.bottom_right_cnn = HairstyleScorePartCNN()

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(2 * (128 * 6 * 10), 512),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.fc_final = nn.Linear(512, 1)

    def forward(self, x):
        x_bottom_left = x[:, :, IMG_RESOLUTION // 2:, :3 * IMG_RESOLUTION // 8]
        x_bottom_right = x[:, :, IMG_RESOLUTION // 2:, 5 * IMG_RESOLUTION // 8:]

        x_bottom_left = self.bottom_left_cnn(x_bottom_left)
        x_bottom_right = self.bottom_right_cnn(x_bottom_right)

        # Fully Connected
        final_x = torch.concat([x_bottom_left, x_bottom_right], dim=1)
        final_x = self.fc1(final_x)
        final_x = self.fc_final(final_x)

        return final_x


# Hairstyle (곱슬머리 vs. 직모) CNN 모델 정의
# Create Date : 2025.05.27
# Last Update Date : 2025.05.27
# - Optimizer 를 Cosine Annealing 에서 Exponential 로 변경

# Arguments:
# - device (device) : CNN 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - cnn_model (nn.Module) : 학습할 CNN 모델

def define_cnn_model(device):
    cnn_model = HairstyleScoreCNN()
    cnn_model.optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.00005)
    cnn_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cnn_model.optimizer, gamma=0.95)

    cnn_model.to(device)
    cnn_model.device = device

    # print summary of CNN model before training
    summary(cnn_model, input_size=(TRAIN_BATCH_SIZE, 3, IMG_RESOLUTION, IMG_RESOLUTION))

    return cnn_model


# Property Score 포맷의 데이터를 Concatenate 하여 PyTorch 형식으로 변환
# Create Date : 2025.05.27
# Last Update Date : -

def concatenate_property_scores(raw_data):
    concatenated_labels = torch.concat([raw_data['label']['hairstyle']])
    concatenated_labels = torch.reshape(concatenated_labels, (1, -1))
    concatenated_labels = torch.transpose(concatenated_labels, 0, 1)
    concatenated_labels = concatenated_labels.to(torch.float32)

    return concatenated_labels


# CNN 모델의 Train Step
# Create Date : 2025.05.27
# Last Update Date : -

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
# Create Date : 2025.05.27
# Last Update Date : 2025.05.27
# - current_epoch 의 값이 음수이면 로깅하지 않는 로직 추가

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

    valid_log = {'epoch': current_epoch, 'hairstyle_score_loss': 0.0, 'hairstyle_score_abs_diff': 0.0}

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
            valid_log[f'hairstyle_score_{metric_name}'] /= total

    # compute corr-coef
    valid_log['hairstyle_score_corr'] = np.corrcoef(np.array(outputs_list)[:, 0], np.array(labels_list)[:, 0])[0][1]

    # save output list and labels list (for first 30 samples in valid dataset)
    if current_epoch >= 0:
        save_valid_output_and_labels(outputs_list, labels_list, img_path_list, current_epoch)

    return valid_log


# Valid Output 및 Label 의 리스트를 csv 형태로 저장 (property_score_cnn/train_logs/property_score_cnn_valid_log.csv)
# Create Date : 2025.05.27
# Last Update Date : -

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
        cnn_valid_log[f'hairstyle_score_{value_type}'] += list(value_list[:, 0])

    cnn_valid_log_path = f'{train_log_dir_path}/property_score_cnn_valid_log.csv'
    pd.DataFrame(cnn_valid_log).to_csv(cnn_valid_log_path, index=False)


# CNN 모델의 Valid Step 에서 상세한 Valid Loss 결과 (Loss, Abs Diff) 저장
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - outputs   (PyTorch Tensor) : Valid dataset 에 대한 predicted output
# - labels    (PyTorch Tensor) : Valid dataset 에 대한 label
# - valid_log (dict)           : CNN 모델의 Validation Log
#                                {'epoch': int, 'valid_loss': float, 'hairstyle_score_loss': float,
#                                 'hairstyle_score_abs_diff': float, 'hairstyle_score_corr': float}

def compute_detailed_valid_results(outputs, labels, valid_log):

    # compute loss
    hairstyle_score_loss = float(cnn_loss_func(outputs[:, :1], labels[:, :1]).detach().cpu().numpy())
    valid_log['hairstyle_score_loss'] += hairstyle_score_loss * labels.size(0)

    # compute abs diff
    hairstyle_score_abs_diff = float(nn.L1Loss()(outputs[:, :1], labels[:, :1]).detach().cpu().numpy())
    valid_log['hairstyle_score_abs_diff'] += hairstyle_score_abs_diff * labels.size(0)


# CNN 모델 학습 (Property Score 계산용)
# Create Date : 2025.05.27
# Last Update Date : 2025.05.27
# - 인수 이름 변경
# - best epoch CNN 정상 loading 확인을 위해 valid dataloader 반환

# Arguments:
# - device                     (device)     : CNN 모델을 mapping 시킬 device (GPU 등)
# - hairstyle_score_dataloader (DataLoader) : 데이터셋의 Data Loader

# Returns:
# - best_epoch_model (nn.Module)  : 학습된 CNN 모델 (valid loss 가 가장 작은 best epoch)
# - valid_loader     (DataLoader) : 데이터셋의 Valid Data Loader

def train_cnn_model(device, hairstyle_score_dataloader):
    cnn_model = define_cnn_model(device)

    # split dataset
    dataset_size = len(hairstyle_score_dataloader.dataset)
    train_size = int(dataset_size * 0.9)
    valid_size = dataset_size - train_size

    train_dataset, valid_dataset = random_split(hairstyle_score_dataloader.dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    # prepare training
    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model = None

    performance_dict = {'epoch': [], 'valid_loss': [],
                        'hairstyle_score_loss': [], 'hairstyle_score_abs_diff': [], 'hairstyle_score_corr': []}

    # run training until early stopping
    while True:

        # run train and validation
        train_cnn_train_step(cnn_model=cnn_model,
                             cnn_train_dataloader=train_loader)

        valid_log = train_cnn_valid_step(cnn_model=cnn_model,
                                         cnn_valid_dataloader=valid_loader,
                                         current_epoch=current_epoch)

        val_loss = valid_log['valid_loss']
        print(f'epoch : {current_epoch}, valid_loss : {val_loss:.4f}')

        # update log
        for k, v in valid_log.items():
            if k == 'epoch':
                performance_dict[k].append(v)
            else:
                performance_dict[k].append(round(v, 4))

        # save train log
        train_log_path = f'{train_log_dir_path}/property_score_cnn_train_log.csv'
        performance_df = pd.DataFrame(performance_dict)
        performance_df.to_csv(train_log_path, index=False)

        # update scheduler
        cnn_model.scheduler.step()

        # update best epoch
        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = current_epoch

            best_epoch_model = HairstyleScoreCNN().to(cnn_model.device)
            best_epoch_model.load_state_dict(cnn_model.state_dict())

        if current_epoch - min_val_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return best_epoch_model, valid_loader


# 데이터셋 (CNN 에 의해 계산된 hairstyle property scores) 의 Data Loader 로딩
# Create Date : 2025.05.27
# Last Update Date : 2025.05.27
# - 함수 이름 변경

# Arguments:
# - 없음

# Returns:
# - hairstyle_score_dataloader (DataLoader) : hairstyle scores 데이터셋의 Data Loader

def get_dataloader():
    all_scores_dir_path = f'{PROJECT_DIR_PATH}/property_score_cnn/segmentation/property_score_results'
    property_score_csv_path = f'{all_scores_dir_path}/all_scores_ohlora_v3.csv'
    property_score_df = pd.read_csv(property_score_csv_path)

    dataset = PropertyScoreImageDataset(dataset_df=property_score_df, transform=stylegan_transform)
    hairstyle_score_dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    return hairstyle_score_dataloader


# best epoch (with min val loss) CNN model 이 바르게 loading 되었는지 체크
# Create Date: 2025.05.27
# Last Update Date: -

# Arguments:
# - cnn_model    (nn.Module)  : 학습된 CNN 모델
# - valid_loader (DataLoader) : 데이터셋의 Valid Data Loader

def check_best_epoch_model_correctly_loaded(cnn_model, valid_loader):
    valid_log = train_cnn_valid_step(cnn_model=cnn_model,
                                     cnn_valid_dataloader=valid_loader,
                                     current_epoch=-1)

    val_loss = valid_log['valid_loss']
    print(f'checked valid loss : {val_loss}')


if __name__ == '__main__':
    cnn_save_path = f'{PROJECT_DIR_PATH}/property_score_cnn/models/ohlora_v3_hairstyle_property_cnn.pth'

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training Property Score CNN : {device}')

    # load DataLoader
    hairstyle_score_dataloader = get_dataloader()

    # train Property Score (hairstyle) CNN
    cnn_model, valid_loader = train_cnn_model(device, hairstyle_score_dataloader)

    # check best epoch model correctly loaded
    cnn_model.device = device
    cnn_model.to(device)
    check_best_epoch_model_correctly_loaded(cnn_model, valid_loader)

    # save model
    torch.save(cnn_model.state_dict(), cnn_save_path)
