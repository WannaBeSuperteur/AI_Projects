import os

from torch.utils.data import DataLoader

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

import stylegan_modified.stylegan_generator as modified_gen
import stylegan_modified.stylegan_generator_inference as modified_inf
from stylegan_modified.stylegan_generator_v2 import run_fine_tuning

from run_stylegan_fine_tuning import (IMAGE_RESOLUTION,
                                      PROPERTY_SCORE_DIR_PATH,
                                      TRAIN_BATCH_SIZE,
                                      ORIGINAL_HIDDEN_DIMS_Z,
                                      PROPERTY_DIMS_Z)

from run_stylegan_fine_tuning import (PropertyScoreImageDataset,
                                      stylegan_transform,
                                      save_model_structure_pdf,
                                      freeze_generator_layers,
                                      print_summary)

import torch
import pandas as pd


# 기존 Pre-train 된 StyleGAN 모델의 state_dict 로딩
# Create Date : 2025.04.13
# Last Update Date : 2025.04.15
# - 불필요한 generator 반환값 삭제 및 필요한 device 인수 추가

# Arguments:
# - device (device) : StyleGAN-FineTune-v2 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict

def load_existing_stylegan_state_dict(device):
    model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v1.pth'

    # load generator state dict
    generator_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    return generator_state_dict


# StyleGAN-FineTune-v1 의 Generator 모델 생성 (with Pre-trained weights of StyleGAN-FineTune-v1)
# Create Date : 2025.04.13
# Last Update Date : 2025.04.15
# - 필요한 device 인수 추가

# Arguments:
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - device               (device)      : StyleGAN-FineTune-v2 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator

def create_stylegan_finetune_v1(generator_state_dict, device):

    # define model
    restructured_generator = modified_gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)

    # set optimizer and scheduler
    restructured_generator.optimizer = torch.optim.AdamW(restructured_generator.parameters(), lr=0.0001)
    restructured_generator.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=restructured_generator.optimizer,
        T_max=10,
        eta_min=0)

    # load state dict
    del generator_state_dict['mapping.dense0.weight']  # size mismatch because of added property vector
    restructured_generator.load_state_dict(generator_state_dict, strict=False)

    # map to device
    restructured_generator.to(device)
    return restructured_generator


# StyleGAN-Fine-Tune-v1 -> v2 로의 Fine-Tuning 이전 inference test 실시
# Create Date : 2025.04.13
# Last Update Date : 2025.04.15
# - 인수 이름 수정 : reconstructed_generator -> generator

# Arguments:
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator

# Returns:
# - stylegan_modified/inference_test_before_finetuning_v2 에 생성 결과 저장

def run_inference_test_before_finetuning(generator):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_before_finetuning_v2'
    modified_inf.synthesize(generator, num=20, save_dir=img_save_dir, z=None, label=None)


# StyleGAN Fine-Tuning 실시 (핵심 속성 값 7개를 latent vector 에 추가)
# Create Date : 2025.04.13
# Last Update Date : 2025.04.15
# - 불필요한 generator 인수 삭제 및 필요한 device 인수 추가

# Arguments:
# - generator_state_dict   (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - fine_tuning_dataloader (DataLoader)  : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - stylegan_modified/stylegan_gen_fine_tuned_v2.pth     에 Fine-Tuning 된 StyleGAN-FineTune-v2 의 Generator 모델 저장
# - stylegan_modified/stylegan_gen_fine_tuned_v2_cnn.pth 에 Fine-Tuning 된 StyleGAN-FineTune-v2 중 CNN 모델 저장

def run_stylegan_fine_tuning(generator_state_dict, fine_tuning_dataloader, device):

    # StyleGAN-FineTune-v1 모델 생성
    generator = create_stylegan_finetune_v1(generator_state_dict, device)

    # 모델 구조를 PDF 로 저장
    save_model_structure_pdf(generator,
                             model_name='stylegan_finetune_v1_generator',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)])

    # StyleGAN-FineTune-v1 모델의 레이어 freeze 처리
    freeze_generator_layers(generator)
    print_summary(generator,
                  model_name='stylegan_finetune_v1_generator',
                  input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                              (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                  print_frozen=True)

    # fine tuning 이전 inference 테스트
    run_inference_test_before_finetuning(generator)

    # fine tuning 실시
    fine_tuned_generator, fine_tuned_generator_cnn, exist_dict = run_fine_tuning(generator, fine_tuning_dataloader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    if not exist_dict['stylegan_finetune_v2']:
        torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v2.pth')
        print('StyleGAN-FineTune-v2 saved')

    if not exist_dict['cnn']:
        torch.save(fine_tuned_generator_cnn.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v2_cnn.pth')
        print('CNN for StyleGAN-FineTune-v2 saved')


# StyleGAN Fine-Tuning 용 데이터셋의 Data Loader 로딩
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - stylegan_ft_loader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

def get_stylegan_fine_tuning_dataloader():
    property_score_csv_path = f'{PROPERTY_SCORE_DIR_PATH}/all_scores_v2.csv'
    property_score_df = pd.read_csv(property_score_csv_path)

    stylegan_ft_dataset = PropertyScoreImageDataset(dataset_df=property_score_df, transform=stylegan_transform)
    stylegan_ft_loader = DataLoader(stylegan_ft_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    return stylegan_ft_loader


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v2 : {device}')

    # load Pre-trained StyleGAN state_dict
    generator_state_dict = load_existing_stylegan_state_dict(device)

    # load DataLoader
    fine_tuning_dataloader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(generator_state_dict, fine_tuning_dataloader, device)
