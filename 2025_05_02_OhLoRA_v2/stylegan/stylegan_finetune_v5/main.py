import stylegan_common.stylegan_generator as gen
import stylegan_common.stylegan_discriminator as dis
import stylegan_common.stylegan_generator_inference as infer
from stylegan_finetune_v5.run_fine_tuning import run_fine_tuning

from common import (get_stylegan_fine_tuning_dataloader,
                    print_summary,
                    save_model_structure_pdf,
                    load_existing_stylegan_finetune_v1)

import torch
import os


PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan/model_structure_pdf'

os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)


IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 3           # eyes, mouth, pose
TRAIN_BATCH_SIZE = 16


# StyleGAN Fine-Tuning 을 위한 Generator Layer Freezing
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

def freeze_generator_layers(finetune_v1_generator):

    # freeze 범위 : Z -> W mapping 을 제외한 모든 레이어
    for name, param in finetune_v1_generator.named_parameters():
        if name.split('.')[0] != 'mapping':
            param.requires_grad = False


# StyleGAN Fine-Tuning 을 위한 Discriminator Layer Freezing
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - finetune_v1_discriminator (nn.Module) : StyleGAN-FineTune-v1 모델의 Discriminator

def freeze_discriminator_layers(finetune_v1_discriminator):

    # freeze 범위 : Last Conv. Layer & Final Fully-Connected Layer 를 제외한 모든 레이어
    for name, param in finetune_v1_discriminator.named_parameters():
        if name.split('.')[0] not in ['layer12', 'layer13', 'layer14']:
            param.requires_grad = False


# StyleGAN Fine-Tuning 이전 inference test 실시
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

# Returns:
# - stylegan/stylegan_finetune_v5/inference_test_before_finetuning 에 생성 결과 저장

def run_inference_test_before_finetuning(finetune_v1_generator):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    finetune_v1_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/inference_test_before_finetuning'
    infer.synthesize(finetune_v1_generator, num=50, save_dir=img_save_dir, z=None, label=None)


# StyleGAN-FineTune-v1 모델 생성 (with Pre-trained weights)
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - generator_state_dict     (OrderedDict) : 기존 Pre-train 된 StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : 기존 Pre-train 된 StyleGAN-FineTune-v1 모델의 Discriminator 의 state_dict
# - device                   (Device)      : CUDA or CPU device

# Returns:
# - finetune_v1_generator     (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator
# - finetune_v1_discriminator (nn.Module) : StyleGAN-FineTune-v1 모델의 Discriminator

def create_stylegan_finetune_v1(generator_state_dict, discriminator_state_dict, device):

    # define model
    finetune_v1_generator = gen.StyleGANGeneratorForV1(resolution=IMAGE_RESOLUTION)
    finetune_v1_discriminator = dis.StyleGANDiscriminator(resolution=IMAGE_RESOLUTION)

    # set optimizer and scheduler
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

    # load state dict
    del generator_state_dict['mapping.dense0.weight']  # size mismatch because of added property vector
    del generator_state_dict['mapping.label_weight']  # size mismatch because of property score size mismatch (7 vs. 3)
    finetune_v1_generator.load_state_dict(generator_state_dict, strict=False)

    del discriminator_state_dict['layer14.weight']  # size mismatch because of added property vector
    del discriminator_state_dict['layer14.bias']    # size mismatch because of added property vector
    finetune_v1_discriminator.load_state_dict(discriminator_state_dict, strict=False)

    # map to device
    finetune_v1_generator.to(device)
    finetune_v1_discriminator.to(device)

    return finetune_v1_generator, finetune_v1_discriminator


# StyleGAN Fine-Tuning 실시 (핵심 속성 값 3개를 latent vector 에 추가)
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - generator_state_dict     (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Discriminator 의 state_dict
# - stylegan_ft_loader       (DataLoader)  : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - device                   (Device)      : CUDA or CPU device

# Returns:
# - stylegan/stylegan_models/stylegan_gen_fine_tuned.pth 에 StyleGAN-FineTune-v5 의 Generator 모델 저장
# - stylegan/stylegan_models/stylegan_dis_fine_tuned.pth 에 StyleGAN-FineTune-v5 의 Discriminator 모델 저장

def run_stylegan_fine_tuning(generator_state_dict, discriminator_state_dict, stylegan_ft_loader, device):

    # StyleGAN-FineTune-v1 모델 생성
    finetune_v1_generator, finetune_v1_discriminator = create_stylegan_finetune_v1(generator_state_dict,
                                                                                   discriminator_state_dict,
                                                                                   device)

    # StyleGAN-FineTune-v1 모델의 레이어 freeze 처리
    freeze_generator_layers(finetune_v1_generator)
    freeze_discriminator_layers(finetune_v1_discriminator)

    # freeze 후 모델 summary 출력
    save_model_structure_pdf(finetune_v1_generator,
                             model_name='finetune_v1_generator (AFTER FREEZING)',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    save_model_structure_pdf(finetune_v1_discriminator,
                             model_name='finetune_v1_discriminator (AFTER FREEZING)',
                             input_size=[(TRAIN_BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    # fine tuning 이전 inference 테스트
    run_inference_test_before_finetuning(finetune_v1_generator)

    # fine tuning 실시
    fine_tuned_generator, fine_tuned_discriminator = run_fine_tuning(finetune_v1_generator,
                                                                     finetune_v1_discriminator,
                                                                     stylegan_ft_loader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_models'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v5.pth')
    torch.save(fine_tuned_discriminator.state_dict(), f'{fine_tuned_model_path}/stylegan_dis_fine_tuned_v5.pth')


def main():

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN : {device}')

    # load Pre-trained StyleGAN
    gen_state_dict, dis_state_dict = load_existing_stylegan_finetune_v1(device)

    # load DataLoader
    stylegan_ft_loader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(gen_state_dict, dis_state_dict, stylegan_ft_loader, device)
