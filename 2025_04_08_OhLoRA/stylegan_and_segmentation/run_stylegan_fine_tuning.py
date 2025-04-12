import stylegan.stylegan_generator as original_gen
import stylegan.stylegan_discriminator as original_dis

import stylegan_modified.stylegan_generator as modified_gen
import stylegan_modified.stylegan_discriminator as modified_dis
import stylegan_modified.stylegan_generator_inference as modified_inf
from stylegan_modified.fine_tuning import run_fine_tuning

import torch
import torch.nn as nn
from torchinfo import summary
from torchview import draw_graph

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd

import os


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/model_structure_pdf'
PROPERTY_SCORE_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/property_score_results'

os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)


IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 7           # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

TRAIN_BATCH_SIZE = 16

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

        self.eyes_scores = dataset_df['eyes_score'].tolist()
        self.hair_color_scores = dataset_df['hair_color_score'].tolist()
        self.hair_length_scores = dataset_df['hair_length_score'].tolist()
        self.mouth_scores = dataset_df['mouth_score'].tolist()
        self.pose_scores = dataset_df['pose_score'].tolist()
        self.background_mean_scores = dataset_df['background_mean_score'].tolist()
        self.background_std_scores = dataset_df['background_std_score'].tolist()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)

        eyes_score = self.eyes_scores[idx]
        hair_color_score = self.hair_color_scores[idx]
        hair_length_score = self.hair_length_scores[idx]
        mouth_score = self.mouth_scores[idx]
        pose_score = self.pose_scores[idx]
        background_mean_score = self.background_mean_scores[idx]
        background_std_score = self.background_std_scores[idx]

        property_scores = {'eyes': eyes_score,
                           'hair_color': hair_color_score,
                           'hair_length': hair_length_score,
                           'mouth': mouth_score,
                           'pose': pose_score,
                           'background_mean': background_mean_score,
                           'background_std': background_std_score}

        # normalize
        image = self.transform(image)

        return {'image': image, 'label': property_scores}


# StyleGAN Fine-Tuning 용 데이터셋의 Data Loader 로딩
# Create Date : 2025.04.12
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - stylegan_ft_loader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

def get_stylegan_fine_tuning_dataloader():
    property_score_csv_path = f'{PROPERTY_SCORE_DIR_PATH}/all_scores.csv'
    property_score_df = pd.read_csv(property_score_csv_path)

    stylegan_ft_dataset = PropertyScoreImageDataset(dataset_df=property_score_df, transform=stylegan_transform)
    stylegan_ft_loader = DataLoader(stylegan_ft_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    return stylegan_ft_loader


# Model Summary (모델 구조) 출력
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - model               (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator 또는 Discriminator
# - model_name          (str)       : 모델을 나타내는 이름
# - input_size          (tuple)     : 모델에 입력될 데이터의 입력 크기
# - print_layer_details (bool)      : 각 레이어 별 detailed info 출력 여부
# - print_frozen        (bool)      : 각 레이어가 freeze 되었는지의 상태 출력 여부

def print_summary(model, model_name, input_size, print_layer_details=False, print_frozen=False):
    print(f'\n\n==== MODEL SUMMARY : {model_name} ====\n')
    summary(model, input_size=input_size)

    if print_layer_details:
        print(model)

    if print_frozen:
        for name, param in model.named_parameters():
            print(f'layer name = {name:40s}, trainable = {param.requires_grad}')


# 기존 Pre-train 된 StyleGAN 모델의 구조를 PDF 로 내보내기
# Create Date : 2025.04.11
# Last Update Date : 2025.04.12
# - layer detail 출력 여부 옵션 (print_layer_details) 추가

# Arguments:
# - model               (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator 또는 Discriminator
# - model_name          (str)       : 모델을 나타내는 이름
# - input_size          (tuple)     : 모델에 입력될 데이터의 입력 크기
# - print_layer_details (bool)      : 각 레이어 별 detailed info 출력 여부
# - print_frozen        (bool)      : 각 레이어가 freeze 되었는지의 상태 출력 여부

def save_model_structure_pdf(model, model_name, input_size, print_layer_details=False, print_frozen=False):
    model_graph = draw_graph(model, input_size=input_size, depth=5)
    visual_graph = model_graph.visual_graph

    # Model Graph 이미지 저장
    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/{model_name}.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    # Model Summary 출력
    print_summary(model, model_name, input_size, print_layer_details=print_layer_details, print_frozen=print_frozen)


# 기존 Pre-train 된 StyleGAN 모델 로딩
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - pretrained_generator     (nn.Module)   : 기존 Pre-train 된 StyleGAN 모델의 Generator
# - pretrained_discriminator (nn.Module)   : 기존 Pre-train 된 StyleGAN 모델의 Discriminator
# - generator_state_dict     (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator 의 state_dict

def load_existing_stylegan():
    pretrained_generator = original_gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)
    pretrained_discriminator = original_dis.StyleGANDiscriminator(resolution=IMAGE_RESOLUTION)

    model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/stylegan_model.pth'

    # load generator state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    generator_state_dict = state_dict['generator']
    discriminator_state_dict = state_dict['discriminator']

    pretrained_generator.load_state_dict(generator_state_dict)
    pretrained_discriminator.load_state_dict(discriminator_state_dict)

    pretrained_generator.to(device)
    pretrained_discriminator.to(device)

    return pretrained_generator, pretrained_discriminator, generator_state_dict, discriminator_state_dict


# StyleGAN Fine-Tuning 을 위한 Generator Layer Freezing
# Create Date : 2025.04.12
# Last Update Date : -

# Arguments:
# - restructured_generator (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator

def freeze_generator_layers(restructured_generator):

    # freeze 범위 : Z -> W mapping 을 제외한 모든 레이어
    for name, param in restructured_generator.named_parameters():
        if name.split('.')[0] != 'mapping':
            param.requires_grad = False


# StyleGAN Fine-Tuning 을 위한 Discriminator Layer Freezing
# Create Date : 2025.04.12
# Last Update Date : -

# Arguments:
# - restructured_discriminator (nn.Module) : StyleGAN 모델의 새로운 구조의 Discriminator

def freeze_discriminator_layers(restructured_discriminator):

    # freeze 범위 : Last Conv. Layer & Final Fully-Connected Layer 를 제외한 모든 레이어
    for name, param in restructured_discriminator.named_parameters():
        if name.split('.')[0] not in ['layer12', 'layer13', 'layer14']:
            param.requires_grad = False


# 새로운 구조의 Generator 및 Discriminator 모델 생성 (with Pre-trained weights)
# Create Date : 2025.04.12
# Last Update Date : -

# Arguments:
# - generator_state_dict     (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator 의 state_dict

# Returns:
# - restructured_generator     (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator
# - restructured_discriminator (nn.Module) : StyleGAN 모델의 새로운 구조의 Discriminator

def create_restructured_stylegan(generator_state_dict, discriminator_state_dict):

    # define model
    restructured_generator = modified_gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)
    restructured_discriminator = modified_dis.StyleGANDiscriminator(resolution=IMAGE_RESOLUTION)

    # set optimizer and scheduler
    restructured_generator.optimizer = torch.optim.AdamW(restructured_generator.parameters(), lr=0.0001)
    restructured_generator.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=restructured_generator.optimizer,
        T_max=10,
        eta_min=0)

    restructured_discriminator.optimizer = torch.optim.AdamW(restructured_discriminator.parameters(), lr=0.0001)
    restructured_discriminator.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=restructured_discriminator.optimizer,
        T_max=10,
        eta_min=0)

    # load state dict
    del generator_state_dict['mapping.dense0.weight']  # size mismatch because of added property vector
    restructured_generator.load_state_dict(generator_state_dict, strict=False)

    del discriminator_state_dict['layer14.weight']  # size mismatch because of added property vector
    del discriminator_state_dict['layer14.bias']    # size mismatch because of added property vector
    restructured_discriminator.load_state_dict(discriminator_state_dict, strict=False)

    # map to device
    restructured_generator.to(device)
    restructured_discriminator.to(device)

    # freeze 전 모델 summary 출력
    save_model_structure_pdf(restructured_generator,
                             model_name='restructured_generator (BEFORE FREEZING)',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)])

    save_model_structure_pdf(restructured_discriminator,
                             model_name='restructured_discriminator (BEFORE FREEZING)',
                             input_size=[(TRAIN_BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)])

    return restructured_generator, restructured_discriminator


# StyleGAN Fine-Tuning 이전 inference test 실시
# Create Date : 2025.04.12
# Last Update Date : -

# Arguments:
# - restructured_generator (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator

# Returns:
# - stylegan_modified/inference_test_before_finetuning 에 생성 결과 저장

def run_inference_test_before_finetuning(restructured_generator):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    restructured_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_before_finetuning'
    modified_inf.synthesize(restructured_generator, num=20, save_dir=img_save_dir, z=None, label=None)


# StyleGAN Fine-Tuning 실시 (핵심 속성 값 7개를 latent vector 에 추가)
# Create Date : 2025.04.12
# Last Update Date : -

# Arguments:
# - pretrained_generator     (nn.Module)   : 기존 Pre-train 된 StyleGAN 모델의 Generator
# - pretrained_discriminator (nn.Module)   : 기존 Pre-train 된 StyleGAN 모델의 Discriminator
# - stylegan_ft_loader       (DataLoader)  : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - generator_state_dict     (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator 의 state_dict

# Returns:
# - stylegan_modified/stylegan_gen_fine_tuned.pth 에 Fine-Tuning 된 StyleGAN 의 Generator 모델 저장
# - stylegan_modified/stylegan_dis_fine_tuned.pth 에 Fine-Tuning 된 StyleGAN 의 Discriminator 모델 저장

def run_stylegan_fine_tuning(pretrained_generator, pretrained_discriminator, stylegan_ft_loader,
                             generator_state_dict, discriminator_state_dict):

    # 모델 구조를 PDF 로 저장
    save_model_structure_pdf(pretrained_generator,
                             model_name='original_pretrained_generator',
                             input_size=(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z))

    save_model_structure_pdf(pretrained_discriminator,
                             model_name='original_pretrained_discriminator',
                             input_size=(TRAIN_BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION))

    # restructured StyleGAN 모델 생성
    restructured_generator, restructured_discriminator = create_restructured_stylegan(generator_state_dict,
                                                                                      discriminator_state_dict)

    # restructured StyleGAN 모델의 레이어 freeze 처리
    freeze_generator_layers(restructured_generator)
    freeze_discriminator_layers(restructured_discriminator)

    # freeze 후 모델 summary 출력
    save_model_structure_pdf(restructured_generator,
                             model_name='restructured_generator (AFTER FREEZING)',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    save_model_structure_pdf(restructured_discriminator,
                             model_name='restructured_discriminator (AFTER FREEZING)',
                             input_size=[(TRAIN_BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    # fine tuning 이전 inference 테스트
    run_inference_test_before_finetuning(restructured_generator)

    # fine tuning 실시
    fine_tuned_generator, fine_tuned_discriminator = run_fine_tuning(restructured_generator,
                                                                     restructured_discriminator,
                                                                     stylegan_ft_loader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned.pth')
    torch.save(fine_tuned_discriminator.state_dict(), f'{fine_tuned_model_path}/stylegan_dis_fine_tuned.pth')


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN : {device}')

    # load Pre-trained StyleGAN
    pretrained_gen, pretrained_dis, gen_state_dict, dis_state_dict = load_existing_stylegan()

    # load DataLoader
    stylegan_ft_loader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(pretrained_gen, pretrained_dis, stylegan_ft_loader, gen_state_dict, dis_state_dict)

