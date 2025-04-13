import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

import stylegan_modified.stylegan_generator as modified_gen
import stylegan_modified.stylegan_generator_inference as modified_inf
from stylegan_modified.stylegan_generator_v2 import run_fine_tuning

from run_stylegan_fine_tuning import IMAGE_RESOLUTION, TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z, PROPERTY_DIMS_Z
from run_stylegan_fine_tuning import (get_stylegan_fine_tuning_dataloader,
                                      save_model_structure_pdf,
                                      freeze_generator_layers,
                                      print_summary)

import torch


# 기존 Pre-train 된 StyleGAN 모델 로딩
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - generator            (nn.Module)   : StyleGAN-FineTune-v1 모델의 Generator
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict

def load_existing_stylegan():
    generator = modified_gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)

    model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v1.pth'

    # load generator state dict
    generator_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    generator.load_state_dict(generator_state_dict)
    generator.to(device)

    return generator, generator_state_dict


# StyleGAN-FineTune-v1 의 Generator 모델 생성 (with Pre-trained weights of StyleGAN-FineTune-v1)
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - generator_state_dict (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict

# Returns:
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator

def create_stylegan_finetune_v1(generator_state_dict):

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
# Last Update Date : -

# Arguments:
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator

# Returns:
# - stylegan_modified/inference_test_before_finetuning_v2 에 생성 결과 저장

def run_inference_test_before_finetuning(restructured_generator):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    restructured_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_before_finetuning_v2'
    modified_inf.synthesize(restructured_generator, num=20, save_dir=img_save_dir, z=None, label=None)


# StyleGAN Fine-Tuning 실시 (핵심 속성 값 7개를 latent vector 에 추가)
# Create Date : 2025.04.13
# Last Update Date : -

# Arguments:
# - generator              (nn.Module)   : StyleGAN-FineTune-v1 모델의 Generator
# - generator_state_dict   (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - fine_tuning_dataloader (DataLoader)  : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - stylegan_modified/stylegan_gen_fine_tuned_v2.pth     에 Fine-Tuning 된 StyleGAN-FineTune-v2 의 Generator 모델 저장
# - stylegan_modified/stylegan_gen_fine_tuned_v2_cnn.pth 에 Fine-Tuning 된 StyleGAN-FineTune-v2 중 CNN 모델 저장

def run_stylegan_fine_tuning(generator, generator_state_dict, fine_tuning_dataloader):

    # 모델 구조를 PDF 로 저장
    save_model_structure_pdf(generator,
                             model_name='stylegan_finetune_v1_generator',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)])

    # StyleGAN-FineTune-v1 모델 생성
    generator = create_stylegan_finetune_v1(generator_state_dict)

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
    fine_tuned_generator, fine_tuned_generator_cnn = run_fine_tuning(generator, fine_tuning_dataloader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v2.pth')
    torch.save(fine_tuned_generator_cnn.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v2_cnn.pth')


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v2 : {device}')

    # load Pre-trained StyleGAN
    generator, generator_state_dict = load_existing_stylegan()

    # load DataLoader
    fine_tuning_dataloader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(generator, generator_state_dict, fine_tuning_dataloader)
