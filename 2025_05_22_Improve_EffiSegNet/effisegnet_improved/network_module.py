# Modified Implementation from https://github.com/ivezakis/effisegnet/blob/main/network_module.py

import os
import time

import lightning as L
import torch
import numpy as np
import pandas as pd
import cv2

from hydra.utils import instantiate
from monai import metrics as mm

global log_dict
log_dict = {'tvt_type': [], 'epoch': [], 'dice': [], 'iou': [], 'recall': [], 'precision': [], 'f1_score': [],
            'train time (s)': [], 'inference time (s)': [],
            'train transform time (s)': [], 'inference transform time (s)': []}

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THRESHOLD = 1.0 / (1.0 + np.exp(-0.5))  # = sigmoid(0.5)


# 이미지를 NumPy image 로 변환
# Create Date: 2025.05.24
# Last Update Date: -

# Arguments:
# - item               (Tensor) : Tensor 형태의 이미지, dim = (1, 3, H, W)
# - imagenet_normalize (bool)   : ImageNet normalization 적용 여부

# Returns:
# - item_ (NumPy array) : NumPy array 형태로 변환된 이미지, dim = (H, W, 3)

def convert_to_numpy_img(item, imagenet_normalize):
    item_ = item.detach().cpu().unsqueeze(dim=0)
    item_ = np.array(item_[0])
    item_ = np.transpose(item_, (1, 2, 0))

    if imagenet_normalize:
        item_ = item_ * IMAGENET_STD + IMAGENET_MEAN  # de-normalize

    item_ = item_ * 255.0
    item_ = item_[:, :, ::-1]
    return item_


# NumPy image 로 변환된 이미지를 파일로 저장
# Create Date: 2025.05.24
# Last Update Date: -

# Arguments:
# - image     (NumPy array) : NumPy array 형태로 변환된 이미지, dim = (H, W, 3)
# - save_path (str)         : 이미지 파일로 저장할 경로

def write_img(image, save_path):
    result, image_arr = cv2.imencode(ext='.jpg',
                                     img=image,
                                     params=[cv2.IMWRITE_JPEG_QUALITY, 95])

    if result:
        with open(save_path, mode='w+b') as f:
            image_arr.tofile(f)


class Net(L.LightningModule):
    def __init__(self, model, criterion, optimizer, lr, batch_size, img_size, scheduler=None):
        super().__init__()
        self.model = model
        self.epoch_no = 0
        self.train_time_sec = 0.0
        self.inference_time_sec = 0.0            # for both valid & test
        self.train_transform_time_sec = 0.0
        self.inference_transform_time_sec = 0.0  # for both valid & test

        self.get_dice = mm.DiceMetric(include_background=False)
        self.get_iou = mm.MeanIoU(include_background=False)
        self.get_recall = mm.ConfusionMatrixMetric(
            include_background=False, metric_name="sensitivity"
        )
        self.get_precision = mm.ConfusionMatrixMetric(
            include_background=False, metric_name="precision"
        )

        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.img_size = img_size
        self.scheduler = scheduler
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters(), lr=self.lr)
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": instantiate(self.scheduler, optimizer=optimizer),
                "monitor": "val_loss",
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, transform_time = batch
        self.train_transform_time_sec += np.sum(transform_time.detach().cpu().numpy())

        train_step_start = time.time()
        if self.model.deep_supervision:
            logits, logits_aux = self(x)

            aux_loss = sum(self.criterion(z, y) for z in logits_aux)
            loss = (self.criterion(logits, y) + aux_loss) / (1 + len(logits_aux))
            self.train_time_sec += time.time() - train_step_start

            self.log("train_loss", loss)
            return loss

        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_time_sec += time.time() - train_step_start

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, transform_time = batch
        self.inference_transform_time_sec += np.sum(transform_time.detach().cpu().numpy())

        validation_step_start = time.time()
        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)
        self.inference_time_sec += time.time() - validation_step_start

        loss = self.criterion(logits, y)
        self.log("val_loss", loss)

        preds = (torch.sigmoid(logits) > THRESHOLD).long()
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

        self.visualize_inference_result(x, y, preds, batch_idx=batch_idx, tvt_type='valid')

        return loss

    def test_step(self, batch, batch_idx):
        x, y, transform_time = batch
        self.inference_transform_time_sec += np.sum(transform_time.detach().cpu().numpy())

        test_step_start = time.time()
        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)
        self.inference_time_sec += time.time() - test_step_start

        loss = self.criterion(logits, y)
        self.log("test_loss", loss)

        preds = (torch.sigmoid(logits) > THRESHOLD).long()
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

        self.visualize_inference_result(x, y, preds, batch_idx=batch_idx, tvt_type='test')

        return loss

    def on_validation_epoch_end(self):
        dice = self.get_dice.aggregate().item()
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()

        self.log("val_dice", dice)
        self.log("val_iou", iou)
        self.log("val_recall", recall)
        self.log("val_precision", precision)
        self.log("val_f1", 2 * (precision * recall) / (precision + recall + 1e-8))

        self.log_csv(dice, iou, recall, precision, tvt_type='valid')

        self.get_dice.reset()
        self.get_iou.reset()
        self.get_recall.reset()
        self.get_precision.reset()

        self.epoch_no += 1

    def on_test_epoch_end(self):
        dice = self.get_dice.aggregate().item()
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()

        self.log("test_dice", dice)
        self.log("test_iou", iou)
        self.log("test_recall", recall)
        self.log("test_precision", precision)
        self.log("test_f1", 2 * (precision * recall) / (precision + recall + 1e-8))

        self.log_csv(max(log_dict['dice']), max(log_dict['iou']), 0.0, 0.0, tvt_type='valid_best_value')
        self.log_csv(dice, iou, recall, precision, tvt_type='test')

        self.get_dice.reset()
        self.get_iou.reset()
        self.get_recall.reset()
        self.get_precision.reset()

    def log_csv(self, dice, iou, recall, precision, tvt_type):
        global log_dict

        log_dict['tvt_type'].append(tvt_type)
        log_dict['epoch'].append(self.epoch_no)
        log_dict['dice'].append(round(dice, 4))
        log_dict['iou'].append(round(iou, 4))
        log_dict['recall'].append(round(recall, 4))
        log_dict['precision'].append(round(precision, 4))

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        log_dict['f1_score'].append(round(f1, 4))

        if tvt_type != 'valid_best_value':
            log_dict['train time (s)'].append(round(self.train_time_sec, 4))
            log_dict['inference time (s)'].append(round(self.inference_time_sec, 4))
            log_dict['train transform time (s)'].append(round(self.train_transform_time_sec, 4))
            log_dict['inference transform time (s)'].append(round(self.inference_transform_time_sec, 4))
        else:
            log_dict['train time (s)'].append(0.0)
            log_dict['inference time (s)'].append(0.0)
            log_dict['train transform time (s)'].append(0.0)
            log_dict['inference transform time (s)'].append(0.0)

        log_csv_path = f'{PROJECT_DIR_PATH}/effisegnet_improved/train_log.csv'
        log_df = pd.DataFrame(log_dict)
        log_df.to_csv(log_csv_path)

        if tvt_type != 'valid_best_value':
            self.train_time_sec = 0.0
            self.inference_time_sec = 0.0
            self.train_transform_time_sec = 0.0
            self.inference_transform_time_sec = 0.0

    def visualize_inference_result(self, x, y, preds, batch_idx, tvt_type):
        visualize_path = f'{PROJECT_DIR_PATH}/effisegnet_improved/inference_result/{tvt_type}/{self.epoch_no}'
        os.makedirs(visualize_path, exist_ok=True)

        for idx, (x_item, y_item, pred) in enumerate(zip(x, y, preds)):
            x_item_ = convert_to_numpy_img(x_item, imagenet_normalize=True)
            y_item_ = convert_to_numpy_img(y_item, imagenet_normalize=False)
            pred_ = convert_to_numpy_img(pred, imagenet_normalize=False)

            y_item_1channel = y_item_[:, :, :1]
            pred_1channel = 0.8 * pred_[:, :, :1]
            overlay_y_pred = np.concatenate(
                [y_item_1channel, pred_1channel, np.zeros((self.img_size, self.img_size, 1))],
                axis=2)
            overlay_x_y_pred = 0.55 * x_item_ + 0.45 * overlay_y_pred

            img_no = batch_idx * self.batch_size + idx

            write_img(x_item_, f'{visualize_path}/img_{img_no:04d}_original_x.jpg')
            write_img(overlay_y_pred, f'{visualize_path}/img_{img_no:04d}_overlay_y_pred.jpg')
            write_img(overlay_x_y_pred, f'{visualize_path}/img_{img_no:04d}_overlay_x_y_pred.jpg')

