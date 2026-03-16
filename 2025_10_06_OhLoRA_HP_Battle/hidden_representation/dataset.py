

from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import numpy as np


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/datasets'
NUM_CLASSES = 10


class AutoEncoderImageDataset(Dataset):
    def __init__(self, dataset_df, transform, dataset_name, tvt_type):
        self.img_paths = dataset_df['img_path'].tolist()    # ex: airplane/0001.png
        self.img_nos = dataset_df['img_no'].tolist()
        self.class_idxs = dataset_df['class_idx'].tolist()

        self.transform = transform
        self.dataset_name = dataset_name
        self.tvt_type = tvt_type                            # 'train' or 'test'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = f'{IMAGE_DATA_DIR_PATH}/{self.dataset_name}/{self.tvt_type}/{self.img_paths}'
        image = read_image(img_path)

        # resize and normalize
        image = self.transform(image)
        class_idx = self.class_idxs[idx]
        class_idx_one_hot = np.eye(NUM_CLASSES)[class_idx]

        return image, class_idx_one_hot

