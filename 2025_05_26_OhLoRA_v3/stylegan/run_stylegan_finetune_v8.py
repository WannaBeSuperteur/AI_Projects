import stylegan_common.stylegan_generator as gen
from common import load_existing_stylegan_finetune_v1, save_model_structure_pdf
from generate_dataset.generate import generate_face_images

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd
import torch
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

IMAGE_RESOLUTION = 256
PDF_BATCH_SIZE = 30
TRAIN_BATCH_SIZE = 8
ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS = 7

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# Image Dataset
class StyleGANFineTuneV8TrainDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_paths = dataset_df['img_path'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        image = self.transform(image)
        simplified_img_path = '/'.join(img_path.split('/')[-2:])

        return {'image': image, 'img_path': simplified_img_path}


# StyleGAN-FineTune-v8 Fine Tuning DataLoader 가져오기
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - stylegan_ft_loader (DataLoader) : StyleGAN-FineTune-v8 Fine-Tuning 용 Data Loader

def get_stylegan_fine_tuning_dataloader():
    all_scores_dir_path = f'{PROJECT_DIR_PATH}/property_score_cnn/segmentation/property_score_results'
    property_score_csv_path = f'{all_scores_dir_path}/all_scores_ohlora_v3.csv'
    dataset_df = pd.read_csv(property_score_csv_path, index_col=0)

    stylegan_ft_dataset = StyleGANFineTuneV8TrainDataset(dataset_df, transform=stylegan_transform)
    stylegan_ft_loader = DataLoader(stylegan_ft_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    return stylegan_ft_loader


if __name__ == '__main__':
    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for fine-tuning StyleGAN-FineTune-v1 : {device}')

    finetune_v1_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)

    # try loading StyleGAN-VectorFind-v7 pre-trained model
    generator_state_dict = load_existing_stylegan_finetune_v1(device)
    finetune_v1_generator.load_state_dict(generator_state_dict)
    print('Existing StyleGAN-VectorFind-v1 Generator load successful!! 😊')

    # create model structure PDF and save
    finetune_v1_generator.to(device)
    save_model_structure_pdf(finetune_v1_generator,
                             model_name='finetune_v8_generator',
                             input_size=[(PDF_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (PDF_BATCH_SIZE, ORIGINALLY_PROPERTY_DIMS)])

    # get dataloader
    stylegan_ft_loader = get_stylegan_fine_tuning_dataloader()
    print(stylegan_ft_loader)

    # run StyleGAN-FineTune-v8 Fine Tuning
    # TODO implementation
