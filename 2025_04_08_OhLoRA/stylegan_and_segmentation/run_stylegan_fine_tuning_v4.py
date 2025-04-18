
import torch
import stylegan_modified.stylegan_generator as modified_gen
import stylegan_modified.stylegan_discriminator as modified_dis
import stylegan_modified.stylegan_generator_inference as modified_inf

from run_stylegan_fine_tuning import TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z, IMAGE_RESOLUTION
from run_stylegan_fine_tuning import save_model_structure_pdf, freeze_generator_layers, freeze_discriminator_layers
from run_stylegan_fine_tuning_v3 import get_stylegan_fine_tuning_dataloader
from stylegan_modified.fine_tuning_v4 import run_fine_tuning

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

PROPERTY_DIMS_Z = 3


# 기존 Pre-train 된 StyleGAN 모델의 state_dict 로딩
# Create Date : 2025.04.18
# Last Update Date : -

# Arguments:
# - device (device) : StyleGAN-FineTune-v4 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - generator_state_dict     (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict

def load_existing_stylegan_state_dict(device):
    gen_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v1.pth'
    dis_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_dis_fine_tuned_v1.pth'

    # load generator state dict
    generator_state_dict = torch.load(gen_model_path, map_location=device, weights_only=True)
    discriminator_state_dict = torch.load(dis_model_path, map_location=device, weights_only=True)

    return generator_state_dict, discriminator_state_dict


# 새로운 구조의 Generator 및 Discriminator 모델 생성 (with Pre-trained weights)
# Create Date : 2025.04.18
# Last Update Date : -

# Arguments:
# - generator_state_dict     (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator 의 state_dict

# Returns:
# - restructured_generator     (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator
# - restructured_discriminator (nn.Module) : StyleGAN 모델의 새로운 구조의 Discriminator

def create_restructured_stylegan(generator_state_dict, discriminator_state_dict):

    # define model
    restructured_generator = modified_gen.StyleGANGeneratorForV4(resolution=IMAGE_RESOLUTION)
    restructured_discriminator = modified_dis.StyleGANDiscriminatorForV4(resolution=IMAGE_RESOLUTION)

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
    del generator_state_dict['mapping.label_weight']   # size mismatch because of modified property vector dim (7 -> 3)
    restructured_generator.load_state_dict(generator_state_dict, strict=False)

    del discriminator_state_dict['layer14.weight']  # size mismatch because of added property vector
    del discriminator_state_dict['layer14.bias']    # size mismatch because of added property vector
    restructured_discriminator.load_state_dict(discriminator_state_dict, strict=False)

    return restructured_generator, restructured_discriminator


# StyleGAN Fine-Tuning 이전 inference test 실시
# Create Date : 2025.04.18
# Last Update Date : -

# Arguments:
# - restructured_generator (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator (StyleGAN-FineTune-v4 로 Fine-Tuning 할)

# Returns:
# - stylegan_modified/inference_test_before_finetuning 에 생성 결과 저장

def run_inference_test_before_finetuning(restructured_generator):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    restructured_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_before_finetuning_v4'
    modified_inf.synthesize(restructured_generator, num=20, save_dir=img_save_dir, z=None, label=None)


# StyleGAN-FineTune-v4 Fine-Tuning 실시 (핵심 속성 값 3개를 latent vector 에 추가)
# Create Date : 2025.04.18
# Last Update Date : -

# Arguments:
# - fine_tuning_dataloader   (DataLoader)  : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - generator_state_dict     (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Generator 의 state_dict
# - discriminator_state_dict (OrderedDict) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator 의 state_dict

# Returns:
# - stylegan_modified/stylegan_gen_fine_tuned_v4.pth 에 Fine-Tuning 된 StyleGAN 의 Generator 모델 저장
# - stylegan_modified/stylegan_dis_fine_tuned_v4.pth 에 Fine-Tuning 된 StyleGAN 의 Discriminator 모델 저장

def run_stylegan_fine_tuning(fine_tuning_dataloader, generator_state_dict, discriminator_state_dict, device):

    # restructured StyleGAN 모델 생성
    restructured_generator, restructured_discriminator = create_restructured_stylegan(generator_state_dict,
                                                                                      discriminator_state_dict)

    # map to device
    restructured_generator.to(device)
    restructured_discriminator.to(device)

    # restructured StyleGAN 모델의 레이어 freeze 처리
    freeze_generator_layers(restructured_generator)
    freeze_discriminator_layers(restructured_discriminator)

    # 모델 구조를 PDF 로 저장 및 모델 summary 출력
    save_model_structure_pdf(restructured_generator,
                             model_name='stylegan_finetune_v4_generator',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    save_model_structure_pdf(restructured_discriminator,
                             model_name='stylegan_finetune_v4_discriminator',
                             input_size=[(TRAIN_BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                             print_frozen=True)

    # fine tuning 이전 inference 테스트
    run_inference_test_before_finetuning(restructured_generator)

    # fine tuning 실시
    fine_tuned_generator, fine_tuned_discriminator = run_fine_tuning(restructured_generator,
                                                                     restructured_discriminator,
                                                                     fine_tuning_dataloader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v4.pth')
    torch.save(fine_tuned_discriminator.state_dict(), f'{fine_tuned_model_path}/stylegan_dis_fine_tuned_v4.pth')


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v4 : {device}')

    # load Pre-trained StyleGAN
    generator_state_dict, discriminator_state_dict = load_existing_stylegan_state_dict(device)

    # load DataLoader
    fine_tuning_dataloader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(fine_tuning_dataloader, generator_state_dict, discriminator_state_dict, device)
