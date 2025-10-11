import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split

from torchvision import models
from sklearn import metrics

import numpy as np
import pandas as pd

is_test = False


TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, TEST_BATCH_SIZE = 16, 4, 4
MAX_EPOCHS, EARLY_STOPPING_ROUNDS = 1000, 10


# CNN 모델 optimizer & scheduler & device 설정
# Create Date : 2025.10.11
# Last Update Date : -

# Arguments:
# - cnn_model (nn.Module) : 학습할 CNN 모델
# - optimizer (optimizer) : 학습할 CNN 모델에 적용할 optimizer
# - scheduler (scheduler) : 학습할 CNN 모델에 적용할 learning rate scheduler

def set_cnn_model_config(cnn_model, optimizer, lr_scheduler):
    cnn_model.optimizer = optimizer
    cnn_model.scheduler = lr_scheduler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model.to(device)
    cnn_model.device = device


# CNN 모델 전체 학습 및 테스트 프로세스 실시
# Create Date : 2025.10.11
# Last Update Date : -

# Arguments:
# - cnn_model               (nn.Module)       : 학습할 CNN 모델
# - cnn_model_class         (nn.Module class) : 학습할 CNN 모델의 class
# - cnn_model_backbone_name (str)             : 학습할 CNN 모델의 backbone 이름 (미지정 시 '')
# - num_classes             (int)             : Class 개수
# - train_loader            (DataLoader)      : 학습 DataLoader
# - valid_loader            (DataLoader)      : 검증 DataLoader
# - test_loader             (DataLoader)      : 테스트 DataLoader
# - valid_loss_csv_path     (str)             : validation loss 기록을 저장할 csv path
# - test_cf_matrix_csv_path (str)             : 테스트 결과의 confusion matrix 기록을 저장할 csv path

# Returns:
# - val_loss_list    (list)      : 검증 Loss 리스트
# - test_accuracy    (float)     : 테스트 정확도
# - best_epoch_model (nn.Module) : best epoch 에서의 model

def run_all_process(cnn_model, cnn_model_class, cnn_model_backbone_name, num_classes,
                    train_loader, valid_loader, test_loader, valid_loss_csv_path='', test_cf_matrix_csv_path=''):

    current_epoch = 0
    min_valid_loss_epoch = -1  # Loss-based Early Stopping
    min_valid_loss = None
    best_epoch_model = None
    val_loss_list = []

    while True:
        run_train(model=cnn_model,
                  train_loader=train_loader,
                  device=cnn_model.device)

        valid_accuracy, valid_loss = run_validation(model=cnn_model,
                                                    valid_loader=valid_loader,
                                                    device=cnn_model.device)

        print(f'epoch={current_epoch}, val_acc={valid_accuracy:.6f}, val_loss={valid_loss:.6f}')
        val_loss_list.append(valid_loss)

        if min_valid_loss is None or valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            min_valid_loss_epoch = current_epoch

            if cnn_model_backbone_name == 'resnet18':
                pretrained_model = cnn_model_class(resnet_model=models.resnet18, num_classes=num_classes)
            elif cnn_model_backbone_name == 'resnet34':
                pretrained_model = cnn_model_class(resnet_model=models.resnet34, num_classes=num_classes)
            elif cnn_model_backbone_name == 'resnet50':
                pretrained_model = cnn_model_class(resnet_model=models.resnet50, num_classes=num_classes)
            else:
                pretrained_model = cnn_model_class(num_classes=num_classes)

            best_epoch_model = pretrained_model.to(cnn_model.device)
            best_epoch_model.load_state_dict(cnn_model.state_dict())

        if current_epoch + 1 >= MAX_EPOCHS or current_epoch - min_valid_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    # assert best epoch model accuracy & loss
    checked_valid_accuracy, checked_valid_loss = run_validation(model=best_epoch_model,
                                                                valid_loader=valid_loader,
                                                                device=best_epoch_model.device)

    assert abs(valid_accuracy - checked_valid_accuracy) <= 1e-6
    assert abs(valid_loss - checked_valid_loss) <= 1e-6

    # save valid result csv
    if valid_loss_csv_path != '':
        epoch_list = list(range(len(val_loss_list)))
        valid_result_dict = {
            'epoch': epoch_list,
            'loss': val_loss_list
        }
        valid_result_df = pd.DataFrame(valid_result_dict)
        valid_result_df.to_csv(valid_loss_csv_path)

    # run test
    # validation 과 목적만 다를 뿐 알고리즘은 거의 동일하므로 valid 시 사용한 함수를 그대로 사용
    test_accuracy, _, test_result = run_validation(model=best_epoch_model,
                                                   valid_loader=test_loader,
                                                   device=best_epoch_model.device,
                                                   return_valid_result=True)

    test_cf_matrix = np.zeros((num_classes + 1, num_classes + 1))

    if test_cf_matrix_csv_path != '':
        for i in range(num_classes):
            for j in range(num_classes):
                test_cf_matrix[i][j] = test_result[(test_result['pred_label_idx'] == i) &
                                                   (test_result['true_label_idx'] == j)].count()
            test_cf_matrix[i][num_classes] = np.sum(test_cf_matrix[i][:num_classes])

        for j in range(num_classes):
            test_cf_matrix[num_classes][j] = np.sum(test_cf_matrix[:num_classes, j])

    test_cf_matrix_df = pd.DataFrame(test_cf_matrix)
    test_cf_matrix_df.to_csv(test_cf_matrix_csv_path, index=False)

    return val_loss_list, test_accuracy, best_epoch_model


# 데이터셋 분리
# Create Date : 2025.10.11
# Last Update Date : -

# args :
# - dataset   (Dataset) : train, valid, test 데이터로 분리할 데이터셋
# - tvt_ratio (list)    : train, valid, test 데이터의 비율

# returns :
# - train_loader (DataLoader) : 학습 DataLoader
# - valid_loader (DataLoader) : 검증 DataLoader
# - test_loader  (DataLoader) : 테스트 DataLoader

def split_tvt(dataset, tvt_ratio):
    assert abs(sum(tvt_ratio) - 1.0) <= 1e-6

    dataset_size = len(dataset)
    train_size = int(dataset_size * tvt_ratio[0])
    valid_size = int(dataset_size * tvt_ratio[1])
    test_size = dataset_size - (train_size + valid_size)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader


# 모델 학습 실시
# Create Date : 2025.03.23
# Last Update Date : -

# args :
# - model        (nn.Module)  : 학습할 모델
# - train_loader (DataLoader) : Training Data Loader
# - device       (Device)     : CUDA or CPU device
# - loss_func    (func)       : Loss Function

# returns :
# - train_loss (float) : 모델의 Training Loss

def run_train(model, train_loader, device, loss_func=nn.CrossEntropyLoss(reduction='sum')):
    model.train()
    total = 0
    train_loss_sum = 0.0

    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).to(torch.float32)

        # train 실시
        model.optimizer.zero_grad()
        outputs = model(images).to(torch.float32)

        loss = loss_func(outputs, labels.unsqueeze(1))
        loss.backward()
        model.optimizer.step()

        # test code
        if is_test and idx % 20 == 0:
            print('train idx:', idx)
            print('output:', np.array(outputs.detach().cpu()).flatten())
            print('label:', np.array(labels.detach().cpu()))

        train_loss_sum += loss.item()
        total += labels.size(0)

    train_loss = train_loss_sum / total
    return train_loss


# Auto-Encoder 모델 학습 실시
# Create Date : 2025.03.25
# Last Update Date : 2025.03.25
# - 특정 epoch / sample (data loader) index 에 대해, (출력, 이미지, latent vector) 를 출력할 수 있도록 수정

# args :
# - model           (nn.Module)  : 학습할 모델
# - train_loader    (DataLoader) : Training Data Loader
# - device          (Device)     : CUDA or CPU device
# - loss_func       (func)       : Loss Function
# - center_crop     (tuple)      : None 이 아니면 이미지의 가운데 (h, w) 픽셀만을 crop
# - force_test_idxs (list(int))  : test code 처럼 (출력, 이미지, latent vector) 를 출력하는 sample (data loader) index 리스트

# returns :
# - train_loss (float) : 모델의 Training Loss

def run_train_ae(model, train_loader, device, loss_func=nn.MSELoss(reduction='sum'),
                 center_crop=None, force_test_idxs=None):

    model.train()
    total = 0
    train_loss_sum = 0.0

    for idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # train 실시
        model.optimizer.zero_grad()
        decoder_outputs = model(images).to(torch.float32)

        # loss 계산 (center_crop 여부에 따라)
        if center_crop is None:
            images_ = images

        else:
            image_height = images.shape[2]
            image_width = images.shape[3]
            h = center_crop[0]
            w = center_crop[1]

            top = image_height // 2 - h // 2
            bottom = image_height // 2 + h // 2
            left = image_width // 2 - w // 2
            right = image_width // 2 + w // 2

            images_ = images[:, :, top:bottom, left:right]

        loss = loss_func(decoder_outputs, images_)

        loss.backward()
        model.optimizer.step()

        # test code
        if (is_test and idx % 20 == 0) or (force_test_idxs is not None and idx in force_test_idxs):
            latent_vectors = model.encoder(images_).to(torch.float32)

            print('\ntrain idx:', idx)
            print('output:', np.array(decoder_outputs.detach().cpu()).flatten()[:5])
            print('image:', np.array(images_.detach().cpu()).flatten()[:5])
            print('latent:\n', np.array(latent_vectors.detach().cpu())[:5, :5])

        train_loss_sum += loss.item()
        total += images.size(0)

    train_loss = train_loss_sum / total
    return train_loss


# 모델 validation 실시
# Create Date : 2025.03.23
# Last Update Date : 2025.10.11
# - 각 sample 별 예측 결과 반환 추가

# args :
# - model               (nn.Module)  : validation 할 모델
# - valid_loader        (DataLoader) : Validation Data Loader
# - device              (Device)     : CUDA or CPU device
# - loss_func           (func)       : Loss Function
# - return_valid_result (boolean)    : valid result 반환 여부

# returns :
# - val_accuracy (float)            : 모델의 validation 정확도
# - val_loss     (float)            : 모델의 validation loss
# - val_result   (Pandas DataFrame) : 각 sample 에 대한 validation 결과

def run_validation(model, valid_loader, device, loss_func=nn.CrossEntropyLoss(reduction='sum'),
                   return_valid_result=False):

    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0
    valid_result_dict = {'pred_label_idx': [], 'true_label_idx': []}

    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)
            val_loss_batch = loss_func(outputs, labels.unsqueeze(1))
            val_loss_sum += val_loss_batch

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_valid_result:
                for pred, label in zip(predicted, labels):
                    valid_result_dict['pred_label_idx'].append(pred)
                    valid_result_dict['true_label_idx'].append(label)

            # test code
            if is_test and idx % 20 == 0:
                print('valid idx:', idx)
                print('output:', np.array(outputs.detach().cpu()).flatten())
                print('label:', np.array(labels.detach().cpu()))

        # Accuracy 계산
        val_accuracy = correct / total
        val_loss = val_loss_sum / total

    if return_valid_result:  # validation 결과 반환
        val_result = pd.DataFrame(valid_result_dict)
        return val_accuracy, val_loss, val_result
    else:
        return val_accuracy, val_loss


# 모델 validation 실시 (TP, TN, FP, FN 및 각종 성능지표 추가 반환)
# Create Date : 2025.04.10
# Last Update Date : -

# args :
# - model        (nn.Module)  : validation 할 모델
# - valid_loader (DataLoader) : Validation Data Loader
# - device       (Device)     : CUDA or CPU device
# - loss_func    (func)       : Loss Function
# - threshold    (float)      : TP, TN, FP, FN 결정을 위한 threshold

# returns :
# - val_result (dict) : validation result (Accuracy, Loss, 기타 성능지표)

def run_validation_detail(model, valid_loader, device, loss_func=nn.CrossEntropyLoss(reduction='sum'), threshold=0.5):
    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    valid_preds = []
    valid_labels = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)
            val_loss_batch = loss_func(outputs, labels.unsqueeze(1))
            val_loss_sum += val_loss_batch

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # compute TP, TN, FP, and FN
            pred_cpu = list(np.array(outputs.detach().cpu()).flatten())
            label_cpu = list(np.array(labels.detach().cpu()).flatten())

            for pred, label in zip(pred_cpu, label_cpu):
                if pred >= threshold and label >= threshold:
                    tp += 1
                elif pred < threshold and label < threshold:
                    tn += 1
                elif pred >= threshold and label < threshold:
                    fp += 1
                else:
                    fn += 1

                valid_preds.append(pred)
                valid_labels.append(label)

            # test code
            if is_test and idx % 20 == 0:
                print('valid idx:', idx)
                print('output:', np.array(outputs.detach().cpu()).flatten())
                print('label:', np.array(labels.detach().cpu()))

        # Accuracy 계산
        val_accuracy = (tp + tn) / total
        val_recall = '' if tp + fn == 0 else tp / (tp + fn)
        val_precision = '' if tp + fp == 0 else tp / (tp + fp)
        val_f1_score = '' if tp == 0 else 2 * val_recall * val_precision / (val_recall + val_precision)

        val_loss = val_loss_sum / total

    val_auroc = metrics.roc_auc_score(valid_labels, valid_preds)

    val_result = {'val_accuracy': val_accuracy, 'val_loss': val_loss,
                  'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                  'val_recall': val_recall, 'val_precision': val_precision, 'val_f1_score': val_f1_score,
                  'val_auroc': val_auroc}

    return val_result
