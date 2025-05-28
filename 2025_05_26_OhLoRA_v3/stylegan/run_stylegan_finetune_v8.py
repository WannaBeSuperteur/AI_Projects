import stylegan_common.stylegan_generator as gen
import stylegan_common.stylegan_discriminator as dis
from common import load_existing_stylegan_finetune_v1_all, save_model_structure_pdf
from stylegan_finetune_v8.fine_tuning_v8 import run_fine_tuning

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd
import torch
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

IMAGE_RESOLUTION = 256
PDF_BATCH_SIZE = 16
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

    print(f'StyleGAN-FineTune-v8 dataset size : {len(stylegan_ft_dataset)}')

    return stylegan_ft_loader


# StyleGAN-FineTune-v1 Generator, Discriminator 의 Optimizer & Scheduler 설정
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - finetune_v1_generator     (nn.Module) : StyleGAN-FineTune-v1 의 Generator
# - finetune_v1_discriminator (nn.Module) : StyleGAN-FineTune-v1 의 Discriminator

def set_optimizer(finetune_v1_generator, finetune_v1_discriminator):
    finetune_v1_generator.optimizer = torch.optim.AdamW(finetune_v1_generator.parameters(), lr=0.0001)
    finetune_v1_generator.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=finetune_v1_generator.optimizer,
        T_max=10,
        eta_min=0)

    finetune_v1_discriminator.optimizer = torch.optim.AdamW(finetune_v1_discriminator.parameters(), lr=0.0001)
    finetune_v1_discriminator.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=finetune_v1_discriminator.optimizer,
        T_max=10,
        eta_min=0)


# StyleGAN Fine-Tuning 을 위한 Generator Layer Freezing
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

def freeze_generator_layers(finetune_v1_generator):
    trainable_synthesis_layers = ['layer0', 'layer1', 'output0']

    # freeze 범위 : Z -> W mapping & synthesize network 의 4 x 4 를 제외한 모든 레이어
    for name, param in finetune_v1_generator.named_parameters():
        if name.split('.')[0] == 'synthesis' and name.split('.')[1] not in trainable_synthesis_layers:
            param.requires_grad = False


# StyleGAN Fine-Tuning 을 위한 Discriminator Layer Freezing
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - finetune_v1_discriminator (nn.Module) : StyleGAN-FineTune-v1 의 Discriminator

def freeze_discriminator_layers(finetune_v1_discriminator):

    # freeze 범위 : Last Conv. Layer & Final Fully-Connected Layer 를 제외한 모든 레이어
    for name, param in finetune_v1_discriminator.named_parameters():
        if name.split('.')[0] not in ['layer12', 'layer13', 'layer14']:
            param.requires_grad = False


if __name__ == '__main__':
    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for fine-tuning StyleGAN-FineTune-v1 : {device}')

    finetune_v1_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)
    finetune_v1_discriminator = dis.StyleGANDiscriminator(resolution=IMAGE_RESOLUTION)

    # load StyleGAN-VectorFind-v1 pre-trained model
    generator_state_dict, discriminator_state_dict = load_existing_stylegan_finetune_v1_all(device)

    finetune_v1_generator.load_state_dict(generator_state_dict)
    print('Existing StyleGAN-VectorFind-v1 Generator load successful!! 😊')

    finetune_v1_discriminator.load_state_dict(discriminator_state_dict)
    print('Existing StyleGAN-VectorFind-v1 Discriminator load successful!! 😊')

    # freeze layers
    freeze_generator_layers(finetune_v1_generator)
    freeze_discriminator_layers(finetune_v1_discriminator)

    # set optimizer
    set_optimizer(finetune_v1_generator, finetune_v1_discriminator)

    # create model structure PDF and save
    finetune_v1_generator.to(device)
    save_model_structure_pdf(finetune_v1_generator,
                             model_name='finetune_v8_generator',
                             input_size=[(PDF_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (PDF_BATCH_SIZE, ORIGINALLY_PROPERTY_DIMS)],
                             print_layer_details=False,
                             print_frozen=True)

    finetune_v1_discriminator.to(device)
    save_model_structure_pdf(finetune_v1_discriminator,
                             model_name='finetune_v8_discriminator',
                             input_size=[(PDF_BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                                         (PDF_BATCH_SIZE, ORIGINALLY_PROPERTY_DIMS)],
                             print_layer_details=False,
                             print_frozen=True)

    # get dataloader
    stylegan_ft_loader = get_stylegan_fine_tuning_dataloader()

    # run StyleGAN-FineTune-v8 Fine Tuning
    fine_tuned_generator, fine_tuned_discriminator = run_fine_tuning(finetune_v1_generator,
                                                                     finetune_v1_discriminator,
                                                                     stylegan_ft_loader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/models'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v8.pth')
    torch.save(fine_tuned_discriminator.state_dict(), f'{fine_tuned_model_path}/stylegan_dis_fine_tuned_v8.pth')
