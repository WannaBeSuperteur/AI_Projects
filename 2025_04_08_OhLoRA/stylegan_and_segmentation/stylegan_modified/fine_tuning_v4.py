# Modified Loss implementation from https://github.com/genforce/genforce/blob/master/runners/losses/logistic_gan_loss.py
# Modified Train Process implementation from https://github.com/genforce/genforce/blob/master/runners/stylegan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_gan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_runner.py
# Train Argument Settings from https://github.com/genforce/genforce/blob/master/configs/stylegan_demo.py


import stylegan_modified.stylegan_generator_inference as modified_inf
from stylegan_modified.fine_tuning import train_step
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 3               # eyes, mouth, pose
TRAIN_BATCH_SIZE = 16
TOTAL_EPOCHS = 500
IMGS_PER_TEST_PROPERTY_SET = 1


def train(generator, generator_smooth, discriminator, stylegan_ft_loader, gen_train_args, dis_train_args,
          r1_gamma, r2_gamma, g_smooth_img):

    """Training function."""
    print('Start training.')

    train_log_dict = {'epoch': [], 'idx': [], 'd_loss': [], 'g_loss': [], 'g_train_count': [],
                      'real_scores_mean': [], 'fake_scores_mean': [], 'real_fake_auroc': []}

    current_epoch = 0

    gen_save_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v4.pth'
    dis_save_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_dis_fine_tuned_v4.pth'
    train_log_save_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/train_log_v4.csv'

    while current_epoch < TOTAL_EPOCHS:
        for idx, raw_data in enumerate(stylegan_ft_loader):
            concatenated_labels = torch.concat([raw_data['label']['eyes'],
                                                raw_data['label']['mouth'],
                                                raw_data['label']['pose']])
            concatenated_labels = torch.reshape(concatenated_labels, (PROPERTY_DIMS_Z, -1))
            concatenated_labels = torch.transpose(concatenated_labels, 0, 1)
            concatenated_labels = concatenated_labels.to(torch.float32)

            data = {
                'image': raw_data['image'].cuda(),
                'label': concatenated_labels.cuda()
            }

            print_result_and_save_image = (idx % 10 == 0 or (current_epoch == 0 and idx < 10))

            d_loss_float, g_loss_float, g_train_count, real_scores_mean, fake_scores_mean, real_fake_auroc =(
                train_step(generator, generator_smooth, discriminator, data,
                           gen_train_args, dis_train_args, r1_gamma, r2_gamma, g_smooth_img,
                           save_image=print_result_and_save_image))

            if print_result_and_save_image:
                print(f'epoch={current_epoch}, idx={idx}, '
                      f'd_loss={d_loss_float:.4f}, g_loss={g_loss_float:.4f}, g_train_count={g_train_count}, '
                      f'real_scores_mean={real_scores_mean:.4f}, fake_scores_mean={fake_scores_mean:.4f}, '
                      f'real_fake_auroc={real_fake_auroc:.4f}')

                run_inference_test_during_finetuning(generator, current_epoch=current_epoch, batch_idx=idx)

                # save train log
                train_log_dict['epoch'].append(current_epoch)
                train_log_dict['idx'].append(idx)
                train_log_dict['d_loss'].append(round(d_loss_float, 4))
                train_log_dict['g_loss'].append(round(g_loss_float, 4))
                train_log_dict['g_train_count'].append(g_train_count)
                train_log_dict['real_scores_mean'].append(round(real_scores_mean, 4))
                train_log_dict['fake_scores_mean'].append(round(fake_scores_mean, 4))
                train_log_dict['real_fake_auroc'].append(round(real_fake_auroc, 4))

                pd.DataFrame(train_log_dict).to_csv(train_log_save_path)

        # save model for EVERY EPOCH
        torch.save(generator.state_dict(), gen_save_path)
        torch.save(discriminator.state_dict(), dis_save_path)

        current_epoch += 1


# StyleGAN Fine-Tuning 중 inference test 실시
# Create Date : 2025.04.18
# Last Update Date : -

# Arguments:
# - restructured_generator (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator
# - current_epoch          (int)       : Fine-Tuning 중 현재 epoch 번호
# - batch_idx              (int)       : Fine-Tuning 중 현재 epoch 에서의 batch index 번호

# Returns:
# - stylegan_modified/inference_test_during_finetuning 에 생성 결과 저장

def run_inference_test_during_finetuning(restructured_generator, current_epoch, batch_idx):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    restructured_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_during_finetuning_v4'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}_idx_{batch_idx:04d}'

    # 생성할 이미지의 Property Label 지정
    mouth_labels = [-1.2, -0.6, 0.0, 0.8, 1.6]
    eyes_labels = [-1.8, 1.8]
    pose_labels = [-1.2, 0.0, 1.2, 2.4, 3.6]

    # 이미지 생성 테스트
    current_idx = 0
    z = np.random.normal(0, 1, size=(IMGS_PER_TEST_PROPERTY_SET, ORIGINAL_HIDDEN_DIMS_Z))

    for mouth_label_idx, mouth_label in enumerate(mouth_labels):
        for eyes_label_idx, eyes_label in enumerate(eyes_labels):
            for pose_label_idx, pose_label in enumerate(pose_labels):
                label = [eyes_label, mouth_label, pose_label]

                label_ = np.array([IMGS_PER_TEST_PROPERTY_SET * [label]])
                label_ = label_.reshape((IMGS_PER_TEST_PROPERTY_SET, PROPERTY_DIMS_Z))

                modified_inf.synthesize(restructured_generator,
                                        num=IMGS_PER_TEST_PROPERTY_SET,
                                        save_dir=img_save_dir,
                                        z=z,
                                        label=label_,
                                        img_name_start_idx=current_idx,
                                        verbose=False)

                current_idx += IMGS_PER_TEST_PROPERTY_SET


# 모델 Fine Tuning 실시
# Create Date : 2025.04.18
# Last Update Date : -

# Arguments:
# - restructured_generator     (nn.Module)  : StyleGAN 모델의 새로운 구조의 Generator
# - restructured_discriminator (nn.Module)  : StyleGAN 모델의 새로운 구조의 Discriminator
# - stylegan_ft_loader         (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator     (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Generator
# - fine_tuned_discriminator (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Discriminator

def run_fine_tuning(restructured_generator, restructured_discriminator, stylegan_ft_loader):

    gen_train_args = dict(w_moving_decay=0.995, style_mixing_prob=0.9,
                          trunc_psi=1.0, trunc_layers=0, randomize_noise=True)
    dis_train_args = dict()

    r1_gamma = 10.0
    r2_gamma = 0.0
    g_smooth_img = 100

    # copy Re-constructed Generator Model
    restructured_generator_smooth = deepcopy(restructured_generator)

    # run Fine-Tuning
    train(restructured_generator, restructured_generator_smooth, restructured_discriminator,
          stylegan_ft_loader, gen_train_args, dis_train_args,
          r1_gamma, r2_gamma, g_smooth_img)

