import stylegan.stylegan_generator as original_gen
import stylegan.stylegan_discriminator as original_dis

import torch
from torchinfo import summary
from torchview import draw_graph

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/model_structure_pdf'
os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
ORIGINAL_HIDDEN_DIMS_Z = 512

TRAIN_BATCH_SIZE = 16


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
# Last Update Date : -

# Arguments:
# - model      (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator 또는 Discriminator
# - model_name (str)       : 모델을 나타내는 이름
# - input_size (tuple)     : 모델에 입력될 데이터의 입력 크기

def save_model_structure_pdf(model, model_name, input_size):
    model_graph = draw_graph(model, input_size=input_size, depth=5)
    visual_graph = model_graph.visual_graph

    # Model Graph 이미지 저장
    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/{model_name}.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    # Model Summary 출력
    print_summary(model, model_name, input_size, print_layer_details=True)


# 기존 Pre-train 된 StyleGAN 모델 로딩
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - pretrained_generator     (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator
# - pretrained_discriminator (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator

def load_existing_stylegan():
    pretrained_generator = original_gen.StyleGANGenerator(resolution=256)
    pretrained_discriminator = original_dis.StyleGANDiscriminator(resolution=256)

    model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/stylegan_model.pth'

    # load generator state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    generator_state_dict = state_dict['generator']
    discriminator_state_dict = state_dict['discriminator']

    pretrained_generator.load_state_dict(generator_state_dict)
    pretrained_discriminator.load_state_dict(discriminator_state_dict)

    pretrained_generator.to(device)
    pretrained_discriminator.to(device)

    return pretrained_generator, pretrained_discriminator


# StyleGAN Fine-Tuning 을 위한 Generator Layer Freezing
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - pretrained_generator (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator

def freeze_generator_layers(pretrained_generator):

    # freeze 범위 : Z -> W mapping 을 제외한 모든 레이어
    for name, param in pretrained_generator.named_parameters():
        if name.split('.')[0] != 'mapping':
            param.requires_grad = False


# StyleGAN Fine-Tuning 을 위한 Discriminator Layer Freezing
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - pretrained_discriminator (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator

def freeze_discriminator_layers(pretrained_discriminator):

    # freeze 범위 : Last Conv. Layer & Final Fully-Connected Layer 를 제외한 모든 레이어
    for name, param in pretrained_discriminator.named_parameters():
        if name.split('.')[0] not in ['layer12', 'layer13', 'layer14']:
            param.requires_grad = False


# 모델 Fine Tuning 실시
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - pretrained_generator     (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator
# - pretrained_discriminator (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator

# Returns:
# - fine_tuned_generator     (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Generator
# - fine_tuned_discriminator (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Discriminator

def run_fine_tuning(pretrained_generator, pretrained_discriminator):
    raise NotImplementedError


# StyleGAN Fine-Tuning 실시 (핵심 속성 값 5개를 latent vector 에 추가)
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - pretrained_generator     (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator
# - pretrained_discriminator (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator

# Returns:
# - stylegan_modified/stylegan_gen_fine_tuned.pth 에 Fine-Tuning 된 StyleGAN 의 Generator 모델 저장
# - stylegan_modified/stylegan_dis_fine_tuned.pth 에 Fine-Tuning 된 StyleGAN 의 Discriminator 모델 저장

def run_stylegan_fine_tuning(pretrained_generator, pretrained_discriminator):

    # 모델 구조를 PDF 로 저장
    save_model_structure_pdf(pretrained_generator,
                             model_name='pretrained_generator',
                             input_size=(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z))

    save_model_structure_pdf(pretrained_discriminator,
                             model_name='pretrained_discriminator',
                             input_size=(TRAIN_BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

    # freeze 처리
    freeze_generator_layers(pretrained_generator)
    freeze_discriminator_layers(pretrained_discriminator)

    # freeze 후 모델 summary 출력
    print_summary(pretrained_generator,
                  model_name='pretrained_generator (AFTER FREEZING)',
                  input_size=(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                  print_frozen=True)

    print_summary(pretrained_discriminator,
                  model_name='pretrained_discriminator (AFTER FREEZING)',
                  input_size=(TRAIN_BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
                  print_frozen=True)

    # fine tuning 실시
    fine_tuned_generator, fine_tuned_discriminator = run_fine_tuning(pretrained_generator, pretrained_discriminator)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned.pth')
    torch.save(fine_tuned_discriminator.state_dict(), f'{fine_tuned_model_path}/stylegan_dis_fine_tuned.pth')


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN : {device}')

    # load Pre-trained StyleGAN
    pretrained_gen, pretrained_dis = load_existing_stylegan()

    # Fine Tuning
    run_stylegan_fine_tuning(pretrained_gen, pretrained_dis)

