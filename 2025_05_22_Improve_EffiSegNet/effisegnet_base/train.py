# Original Implementation from https://github.com/ivezakis/effisegnet/blob/main/train.py

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
from monai.networks.nets.efficientnet import get_efficientnet_image_size

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from common.datamodule import KvasirSEGDataset
from network_module import Net

L.seed_everything(42, workers=True)

os.environ["HYDRA_FULL_ERROR"] = "1"  # for hydra debugging
os.environ["USE_LIBUV"] = "0"         # to prevent "use_libuv was requested but PyTorch was build without libuv support"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_float32_matmul_precision("medium")

# remove warnings
import warnings
warnings.filterwarnings('ignore')


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    logger = loggers.TensorBoardLogger("logs/", name=str(cfg.run_name))

    model = instantiate(cfg.model.object)
    if cfg.img_size == "derived":
        img_size = get_efficientnet_image_size(model.model_name)
    else:
        img_size = cfg.img_size

    dataset = KvasirSEGDataset(batch_size=cfg.batch_size, img_size=img_size)

    net = Net(
        model=model,
        criterion=instantiate(cfg.criterion),
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        img_size=img_size,
        scheduler=cfg.scheduler,
    )

    trainer = instantiate(cfg.trainer, logger=logger)

    # if efficientnetb5, b6, or b7, use binsearch to find the largest batch size
    if cfg.model.object.model_name in ["efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(net, dataset, mode="binsearch")

    trainer.fit(net, dataset)
    trainer.test(net, dataset)


if __name__ == "__main__":
    main()
