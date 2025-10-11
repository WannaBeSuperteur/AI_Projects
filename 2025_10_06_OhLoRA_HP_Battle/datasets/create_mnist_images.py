
import pandas as pd
import numpy as np
import os
from PIL import Image

IMG_SIZE = 28


def save_image(idx, img_np, img_dir_path):

    # convert to RGB image
    img_np = np.reshape(img_np, (IMG_SIZE, IMG_SIZE))
    rgb_img_np = np.dstack([img_np, img_np, img_np])

    # save image
    img_save_path = os.path.join(img_dir_path, f'{idx:04d}.png')
    os.makedirs(img_dir_path, exist_ok=True)

    rgb_img_np = rgb_img_np.astype(np.uint8)
    PIL_image = Image.fromarray(rgb_img_np, 'RGB')
    PIL_image.save(img_save_path)


def create_dataset(dataset_name, dataset_csv_prefix):
    print(f'dataset_name: {dataset_name}')

    os.makedirs(f'{dataset_name}/train', exist_ok=True)
    os.makedirs(f'{dataset_name}/test', exist_ok=True)

    train_csv_path = f'{dataset_name}/{dataset_csv_prefix}_train.csv'
    test_csv_path = f'{dataset_name}/{dataset_csv_prefix}_test.csv'
    train_csv = pd.read_csv(train_csv_path)
    test_csv = pd.read_csv(test_csv_path)

    # save train images
    print('saving train images ...')

    for idx, row in train_csv.iterrows():
        row_ = row.tolist()

        label = row_[0]
        img_dir_path = f'{dataset_name}/train/{label}'
        img_np = np.array(row_[1:])
        save_image(idx, img_np, img_dir_path)

    # save test images
    print('saving test images ...')

    for idx, row in test_csv.iterrows():
        row_ = row.tolist()

        label = row_[0]
        img_dir_path = f'{dataset_name}/test/{label}'
        img_np = np.array(row_[1:])
        save_image(idx, img_np, img_dir_path)


if __name__ == '__main__':
    create_dataset('mnist', 'mnist')
    create_dataset('fashion_mnist', 'fashion-mnist')
