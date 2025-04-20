
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image

import stylegan_modified.stylegan_generator as modified_gen
import stylegan_modified.stylegan_discriminator as modified_dis
import stylegan_modified.stylegan_generator_inference as modified_inf

from run_stylegan_fine_tuning import TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z, IMAGE_RESOLUTION
from run_stylegan_fine_tuning import save_model_structure_pdf, freeze_generator_layers, freeze_discriminator_layers
from run_stylegan_fine_tuning import stylegan_transform
from run_stylegan_fine_tuning_v3 import get_stylegan_fine_tuning_dataloader

from stylegan_modified.stylegan_generator_v2 import load_cnn_model
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


# StyleGAN-FineTune-v4 학습 중 생성한 이미지의 eyes, mouth, pose 의 오차 및 상관계수 (생성된 이미지에 대한 CNN 산출값 vs. 의도한 label 값) 기록 (테스트용)
# Create Date : 2025.04.20
# Last Update Date : -

# Arguments:
# - max_epochs (int)    : 학습 중 생성된 이미지가 있는 최대 epoch 횟수
# - device     (device) : Property CNN 모델을 mapping 시킬 device (GPU 등)

def record_property_score_error_info(max_epochs, device):
    error_info_log = {'epoch': [], 'batch_idx': [],
                      'eyes_corr': [], 'mouth_corr': [], 'pose_corr': [],
                      'eyes_mae': [], 'mouth_mae': [], 'pose_mae': []}

    # load pre-trained CNN model for property
    stylegan_and_seg_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation'
    property_cnn_save_path = f'{stylegan_and_seg_path}/stylegan_modified/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_cnn = load_cnn_model(property_cnn_save_path, device)

    img_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_during_finetuning_v4'

    # intended Property Labels
    eyes_label_types = [-1.8, 1.8]
    mouth_label_types = [-1.2, -0.6, 0.0, 0.8, 1.6]
    pose_label_types = [-1.2, 0.0, 1.2, 2.4, 3.6]

    eyes_labels = []
    mouth_labels = []
    pose_labels = []

    for mouth in mouth_label_types:
        for eyes in eyes_label_types:
            for pose in pose_label_types:
                eyes_labels.append(eyes)
                mouth_labels.append(mouth)
                pose_labels.append(pose)

    label_count = len(eyes_labels)

    # compute MAE and corr-coef for each (epoch, batch_idx) pair
    for epoch in range(max_epochs):
        batch_idx_dirs = list(filter(lambda x: x.startswith(f'epoch_{epoch:04d}') and int(x.split('_')[3]) % 20 == 0,
                                     os.listdir(img_dir)))
        batch_idx_dir_paths = [f'{img_dir}/{batch_idx_dir}' for batch_idx_dir in batch_idx_dirs]

        for batch_idx_dir_path in batch_idx_dir_paths:
            image_names = os.listdir(batch_idx_dir_path)
            image_paths = [f'{batch_idx_dir_path}/{name}' for name in image_names]
            batch_idx = int(batch_idx_dir_path.split('_')[-1])

            print(f'checking epoch {epoch} batch {batch_idx} ...')

            error_info_log['epoch'].append(epoch)
            error_info_log['batch_idx'].append(batch_idx)

            eyes_cnn_scores = []
            mouth_cnn_scores = []
            pose_cnn_scores = []

            for image_name, image_path in zip(image_names, image_paths):
                image = read_image(image_path)
                image = stylegan_transform(image)

                with torch.no_grad():
                    property_scores = property_cnn(image.unsqueeze(0).cuda())
                    property_score_np = property_scores.detach().cpu().numpy()

                    eyes_cnn_scores.append(property_score_np[0][0])
                    mouth_cnn_scores.append(property_score_np[0][3])
                    pose_cnn_scores.append(property_score_np[0][4])

            eyes_corr = np.corrcoef(eyes_labels, eyes_cnn_scores)[0][1]
            mouth_corr = np.corrcoef(mouth_labels, mouth_cnn_scores)[0][1]
            pose_corr = np.corrcoef(pose_labels, pose_cnn_scores)[0][1]

            eyes_mae = sum(abs(eyes_labels[i] - eyes_cnn_scores[i]) for i in range(label_count)) / label_count
            mouth_mae = sum(abs(mouth_labels[i] - mouth_cnn_scores[i]) for i in range(label_count)) / label_count
            pose_mae = sum(abs(pose_labels[i] - pose_cnn_scores[i]) for i in range(label_count)) / label_count

            error_info_log['eyes_corr'].append(eyes_corr)
            error_info_log['mouth_corr'].append(mouth_corr)
            error_info_log['pose_corr'].append(pose_corr)

            error_info_log['eyes_mae'].append(eyes_mae)
            error_info_log['mouth_mae'].append(mouth_mae)
            error_info_log['pose_mae'].append(pose_mae)

        # save as csv
        error_info_log_df = pd.DataFrame(error_info_log)
        error_info_log_df.to_csv(f'{stylegan_and_seg_path}/stylegan_modified/train_log_v4_errors.csv')


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v4 : {device}')

    # check error (after Fine-Tuning for some epochs)
#    record_property_score_error_info(max_epochs=79, device=device)

    # load Pre-trained StyleGAN
    generator_state_dict, discriminator_state_dict = load_existing_stylegan_state_dict(device)

    # load DataLoader
    fine_tuning_dataloader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(fine_tuning_dataloader, generator_state_dict, discriminator_state_dict, device)
