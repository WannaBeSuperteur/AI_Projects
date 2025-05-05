from collections import OrderedDict

import stylegan_common.stylegan_generator as gen
import stylegan_common.discriminator as dis
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
# Create Date : 2025.05.04
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

def freeze_generator_layers(finetune_v1_generator):

    # freeze 범위 : Z -> W mapping 을 제외한 모든 레이어
    for name, param in finetune_v1_generator.named_parameters():
        if name.split('.')[0] != 'mapping':
            param.requires_grad = False


# StyleGAN Fine-Tuning 을 위한 Discriminator Layer Freezing
# Create Date : 2025.05.04
# Last Update Date : -

# Arguments:
# - finetune_v1_discriminator (nn.Module) : StyleGAN-FineTune-v1 모델의 Discriminator

def freeze_discriminator_layers(finetune_v1_discriminator):

    # freeze 범위 : eyes/mouth/pose score CNN 의 Fully-Connected Layer 를 제외한 모든 레이어
    for name, param in finetune_v1_discriminator.named_parameters():
        is_property_score_cnn = name.split('.')[0] in ['eyes_score_cnn', 'mouth_score_cnn', 'pose_score_cnn']
        is_not_fc_layer = len(name.split('.')) >= 2 and 'fc' not in name.split('.')[1]

        if is_property_score_cnn and is_not_fc_layer:
            param.requires_grad = False


# StyleGAN Fine-Tuning 이전 inference test 실시
# Create Date : 2025.05.03
# Last Update Date : 2025.05.05
# - trunc_psi 값 1.0 -> 0.5 로 변경 (for Conditional Generation)

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

# Returns:
# - stylegan/stylegan_finetune_v5/inference_test_before_finetuning 에 생성 결과 저장

def run_inference_test_before_finetuning(finetune_v1_generator):
    kwargs_val = dict(trunc_psi=0.5, trunc_layers=0, randomize_noise=False)
    finetune_v1_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/inference_test_before_finetuning'
    infer.synthesize(finetune_v1_generator, num=50, save_dir=img_save_dir, z=None, label=None)


# StyleGAN-FineTune-v5 Fine-Tuning 할 모델 생성 (with Pre-trained weights)
# Create Date : 2025.05.03
# Last Update Date : 2025.05.04
# - 모델 디렉토리 이름 변경 (stylegan_models -> models) 반영
# - Gender CNN 을 Discriminator 에 추가

# Arguments:
# - generator_state_dict (OrderedDict) : 기존 Pre-train 된 StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - device               (Device)      : CUDA or CPU device

# Returns:
# - finetune_v1_generator     (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator
# - finetune_v1_discriminator (nn.Module) : StyleGAN-FineTune-v1 모델의 Discriminator

def create_stylegan_finetune_v1(generator_state_dict, device):

    # define model
    finetune_v1_generator = gen.StyleGANGeneratorForV5(resolution=IMAGE_RESOLUTION)
    finetune_v1_discriminator = dis.DiscriminatorForV5()

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

    # load state dict (generator)
    del generator_state_dict['mapping.dense0.weight']  # size mismatch because of added property vector
    del generator_state_dict['mapping.label_weight']  # size mismatch because of property score size mismatch (7 vs. 3)
    finetune_v1_generator.load_state_dict(generator_state_dict, strict=False)

    # load state dict (discriminator / property)
    property_cnn_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_score_cnn_state_dict = torch.load(property_cnn_path, map_location=device, weights_only=False)
    finetune_v1_discriminator.load_state_dict(property_score_cnn_state_dict, strict=False)

    # load state dict (discriminator / gender)
    """
    gender_cnn_path = f'{PROJECT_DIR_PATH}/stylegan/models/gender_model_0.pth'
    gender_score_cnn_state_dict = torch.load(gender_cnn_path, map_location=device, weights_only=False)

    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'fc1', 'fc_final']
    gender_score_cnn_state_dict_ = OrderedDict()

    for layer_name in layer_names:
        weight_name = f'{layer_name}.0.weight'
        bias_name = f'{layer_name}.0.bias'

        gender_score_cnn_state_dict_[f'gender_score_cnn.{weight_name}'] = gender_score_cnn_state_dict[weight_name]
        gender_score_cnn_state_dict_[f'gender_score_cnn.{bias_name}'] = gender_score_cnn_state_dict[bias_name]

    finetune_v1_discriminator.load_state_dict(gender_score_cnn_state_dict_, strict=False)
    """

    # map to device
    finetune_v1_generator.to(device)
    finetune_v1_discriminator.to(device)

    return finetune_v1_generator, finetune_v1_discriminator


# StyleGAN Fine-Tuning 실시 (핵심 속성 값 3개를 latent vector 에 추가)
# Create Date : 2025.05.03
# Last Update Date : 2025.05.04
# - 모델 디렉토리 이름 변경 (stylegan_models -> models) 반영

# Arguments:
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - stylegan_ft_loader   (DataLoader)  : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - device               (Device)      : CUDA or CPU device

# Returns:
# - stylegan/models/stylegan_gen_fine_tuned.pth 에 StyleGAN-FineTune-v5 의 Generator 모델 저장
# - stylegan/models/stylegan_dis_fine_tuned.pth 에 StyleGAN-FineTune-v5 의 Discriminator 모델 저장

def run_stylegan_fine_tuning(generator_state_dict, stylegan_ft_loader, device):

    # StyleGAN-FineTune-v1 모델 생성
    finetune_v1_generator, finetune_v1_discriminator = create_stylegan_finetune_v1(generator_state_dict, device)

    # StyleGAN-FineTune-v1 모델의 레이어 freeze 처리
#    freeze_generator_layers(finetune_v1_generator)
#    freeze_discriminator_layers(finetune_v1_discriminator)

    # freeze 후 모델 summary 출력
    save_model_structure_pdf(finetune_v1_generator,
                             model_name='finetune_v1_generator',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    save_model_structure_pdf(finetune_v1_discriminator,
                             model_name='finetune_v1_discriminator',
                             input_size=[(TRAIN_BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    # fine tuning 이전 inference 테스트
    run_inference_test_before_finetuning(finetune_v1_generator)

    # fine tuning 실시
    fine_tuned_generator, fine_tuned_discriminator = run_fine_tuning(finetune_v1_generator,
                                                                     finetune_v1_discriminator,
                                                                     stylegan_ft_loader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/models'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v5.pth')
    torch.save(fine_tuned_discriminator.state_dict(), f'{fine_tuned_model_path}/stylegan_dis_fine_tuned_v5.pth')


def main():

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN : {device}')

    # load Pre-trained StyleGAN
    gen_state_dict = load_existing_stylegan_finetune_v1(device)

    # load DataLoader
    stylegan_ft_loader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(gen_state_dict, stylegan_ft_loader, device)
