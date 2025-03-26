import torch.nn as nn
import torch
import numpy as np  # for test code

is_test = False


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
# Last Update Date : -

# args :
# - model        (nn.Module)  : validation 할 모델
# - valid_loader (DataLoader) : Validation Data Loader
# - device       (Device)     : CUDA or CPU device
# - loss_func    (func)       : Loss Function

# returns :
# - val_accuracy (float) : 모델의 validation 정확도
# - val_loss     (float) : 모델의 validation loss

def run_validation(model, valid_loader, device, loss_func=nn.CrossEntropyLoss(reduction='sum')):
    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)
            val_loss_batch = loss_func(outputs, labels.unsqueeze(1))
            val_loss_sum += val_loss_batch

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # test code
            if is_test and idx % 20 == 0:
                print('valid idx:', idx)
                print('output:', np.array(outputs.detach().cpu()).flatten())
                print('label:', np.array(labels.detach().cpu()))

        # Accuracy 계산
        val_accuracy = correct / total
        val_loss = val_loss_sum / total

    return val_accuracy, val_loss
