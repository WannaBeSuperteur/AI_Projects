# Modified Implementation from https://github.com/ivezakis/effisegnet/blob/main/network_module.py

import os
import lightning as L
import torch
import pandas as pd
from hydra.utils import instantiate
from monai import metrics as mm

global log_dict
log_dict = {'tvt_type': [], 'epoch': [], 'dice': [], 'iou': [], 'recall': [], 'precision': []}

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


class Net(L.LightningModule):
    def __init__(self, model, criterion, optimizer, lr, scheduler=None):
        super().__init__()
        self.model = model
        self.epoch_no = 0

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
        x, y = batch

        if self.model.deep_supervision:
            logits, logits_aux = self(x)

            aux_loss = sum(self.criterion(z, y) for z in logits_aux)
            loss = (self.criterion(logits, y) + aux_loss) / (1 + len(logits_aux))

            self.log("train_loss", loss)
            return loss

        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)

        loss = self.criterion(logits, y)
        self.log("val_loss", loss)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)

        loss = self.criterion(logits, y)
        self.log("test_loss", loss)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

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

        log_csv_path = f'{PROJECT_DIR_PATH}/effisegnet_base/train_log.csv'
        log_df = pd.DataFrame(log_dict)
        log_df.to_csv(log_csv_path)
