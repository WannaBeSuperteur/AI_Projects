# Modified Loss implementation from https://github.com/genforce/genforce/blob/master/runners/losses/logistic_gan_loss.py
# Modified Train Process implementation from https://github.com/genforce/genforce/blob/master/runners/stylegan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_gan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_runner.py
# Train Argument Settings from https://github.com/genforce/genforce/blob/master/configs/stylegan_demo.py


import os
import time

from torchvision.io import read_image

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

import stylegan_common.stylegan_generator_inference as infer
from property_score_cnn import load_cnn_model as load_property_cnn_model
from common import stylegan_transform


ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 3           # eyes, mouth, pose
TRAIN_BATCH_SIZE = 16
TOTAL_EPOCHS = 500
IMGS_PER_TEST_PROPERTY_SET = 10


def compute_grad_penalty(images, scores):
    """Computes gradient penalty."""
    image_grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=images,
        create_graph=True,
        retain_graph=True)[0].view(images.shape[0], -1)
    penalty = image_grad.pow(2).sum(dim=1).mean()
    return penalty


def compute_g_loss(generator, discriminator, data, gen_train_args, save_image):  # pylint: disable=no-self-use
    """Computes loss for generator."""
    # TODO: Use random labels.

    batch_size = data['image'].shape[0]
    labels = data['label']

    latents = torch.randn(batch_size, ORIGINAL_HIDDEN_DIMS_Z).cuda()
    fakes = generator(latents, label=labels, **gen_train_args)['image']
    fake_scores = discriminator(fakes, label=labels)

    if save_image:
        save_real_fake_imgs(fakes)

    mse_loss_eyes = F.mse_loss(fake_scores[:, :1], fake_scores[:, 3:4], reduction='mean')
    mse_loss_mouth = F.mse_loss(fake_scores[:, 1:2], fake_scores[:, 4:5], reduction='mean')
    mse_loss_pose = F.mse_loss(fake_scores[:, 2:3], fake_scores[:, 5:6], reduction='mean')
    g_loss = (mse_loss_eyes + mse_loss_mouth + mse_loss_pose) / 3.0

    return g_loss


# generator     -> layers_to_train = ['mapping']
# discriminator -> layers_to_train = (all layers)
def set_model_requires_grad(model, model_name, requires_grad):
    """Sets the `requires_grad` configuration for a particular model."""

    assert model_name in ['generator', 'discriminator']

    for name, param in model.named_parameters():

        if requires_grad:
            if model_name == 'generator':
                if name.split('.')[0] == 'mapping':
                    param.requires_grad = True

            elif model_name == 'discriminator':
                param.requires_grad = True

        else:
            param.requires_grad = False


def train_step(generator, discriminator, data, gen_train_args, save_image):

    # Update generator.
    set_model_requires_grad(discriminator, 'discriminator', False)
    set_model_requires_grad(generator, 'generator', True)

    g_loss = compute_g_loss(generator, discriminator, data, gen_train_args, save_image)
    generator.optimizer.zero_grad()
    g_loss.backward()
    generator.optimizer.step()

    g_loss_float = float(g_loss.detach().cpu())
    return g_loss_float


def train(generator, discriminator, stylegan_ft_loader, gen_train_args):

    """Training function."""
    print('Start training.')

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v5 : {device}')

    train_log_dict = {'epoch': [], 'idx': [], 'g_loss': [],
                      'eyes_corr': [], 'mouth_corr': [], 'pose_corr': [],
                      'eyes_mae': [], 'mouth_mae': [], 'pose_mae': []}

    current_epoch = 0

    gen_save_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/stylegan_gen_fine_tuned_v5.pth'
    dis_save_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/stylegan_dis_fine_tuned_v5.pth'
    train_log_save_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/train_log.csv'

    property_cnn_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_score_cnn = load_property_cnn_model(property_cnn_path, device)

    while current_epoch < TOTAL_EPOCHS:
        for idx, raw_data in enumerate(stylegan_ft_loader):
            concatenated_labels = concatenate_property_scores(raw_data)

            data = {
                'image': raw_data['image'].cuda(),
                'label': concatenated_labels.cuda()
            }
            print_result_and_save_image = (idx % 10 == 0 or (current_epoch == 0 and idx < 10))
            g_loss_float = train_step(generator, discriminator, data, gen_train_args,
                                      save_image=print_result_and_save_image)

            if print_result_and_save_image:
                print(f'epoch={current_epoch}, idx={idx}, g_loss={g_loss_float:.4f}')

                corr_mae_dict = run_inference_test_during_finetuning(generator,
                                                                     current_epoch=current_epoch,
                                                                     batch_idx=idx,
                                                                     property_score_cnn=property_score_cnn)

                # save train log
                train_log_dict['epoch'].append(current_epoch)
                train_log_dict['idx'].append(idx)
                train_log_dict['g_loss'].append(round(g_loss_float, 4))

                corr_mae_keys = ['eyes_corr', 'mouth_corr', 'pose_corr', 'eyes_mae', 'mouth_mae', 'pose_mae']

                for corr_mae_key in corr_mae_keys:
                    train_log_dict[corr_mae_key].append(round(corr_mae_dict[corr_mae_key], 4))

                pd.DataFrame(train_log_dict).to_csv(train_log_save_path)

        # save model for EVERY EPOCH
        torch.save(generator.state_dict(), gen_save_path)
        torch.save(discriminator.state_dict(), dis_save_path)

        current_epoch += 1


# Property Score 포맷의 데이터를 Concatenate 하여 PyTorch 형식으로 변환
# Create Date : 2025.05.03
# Last Update Date : -

def concatenate_property_scores(raw_data):
    concatenated_labels = torch.concat([raw_data['label']['eyes'],
                                        raw_data['label']['mouth'],
                                        raw_data['label']['pose']])

    concatenated_labels = torch.reshape(concatenated_labels, (PROPERTY_DIMS_Z, -1))
    concatenated_labels = torch.transpose(concatenated_labels, 0, 1)
    concatenated_labels = concatenated_labels.to(torch.float32)

    return concatenated_labels


# 모델의 각 레이어의 trainable / fronzen 상태 확인
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - check_id      (int)       : 여러 번 check 할 때, 각 check 하는 시점을 구분하기 위한 ID
# - generator     (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator
# - discriminator (nn.Module) : StyleGAN 모델의 새로운 구조의 Discriminator

def check_model_trainable_status(check_id, generator, discriminator):
    for name, param in generator.named_parameters():
        print(f'({check_id}) generator layer name = {name}, trainable = {param.requires_grad}')

    for name, param in discriminator.named_parameters():
        print(f'({check_id}) discriminator layer name = {name}, trainable = {param.requires_grad}')


# StyleGAN Fine-Tuning 중 inference test 실시
# Create Date : 2025.05.03
# Last Update Date : 2025.05.04
# - train_log 에 추가할 각 핵심 속성 값 별 Corr-coef, MAE 반환값 추가

# Arguments:
# - restructured_generator (nn.Module) : StyleGAN 모델의 새로운 구조의 Generator
# - current_epoch          (int)       : Fine-Tuning 중 현재 epoch 번호
# - batch_idx              (int)       : Fine-Tuning 중 현재 epoch 에서의 batch index 번호
# - property_score_cnn     (nn.Module) : 핵심 속성 값을 계산하기 위한 CNN

# Returns:
# - corr_mae_dict (dict) : train_log 에 추가할 각 핵심 속성 값 별 Corr-coef 및 MAE
# - stylegan_modified/inference_test_during_finetuning 에 생성 결과 및 각 이미지 별 Property Score CNN 예측 핵심 속성 값 저장

def run_inference_test_during_finetuning(restructured_generator, current_epoch, batch_idx, property_score_cnn):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    restructured_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/inference_test_during_finetuning'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}_idx_{batch_idx:04d}'

    # label: 'eyes', 'mouth', 'pose'
    current_idx = 0

    eyes_cnn_scores = []
    mouth_cnn_scores = []
    pose_cnn_scores = []
    eyes_labels = []
    mouth_labels = []
    pose_labels = []

    z = np.random.normal(0, 1, size=(IMGS_PER_TEST_PROPERTY_SET, ORIGINAL_HIDDEN_DIMS_Z))

    labels = [[-1.6, -1.2, -1.2],
              [-1.6, -1.2,  2.4],
              [-1.6,  1.6, -1.2],
              [-1.6,  1.6,  2.4],
              [ 1.6, -1.2, -1.2],
              [ 1.6, -1.2,  2.4],
              [ 1.6,  1.6, -1.2],
              [ 1.6,  1.6,  2.4]]

    for label in labels:
        label_ = np.array([IMGS_PER_TEST_PROPERTY_SET * [label]])
        label_ = label_.reshape((IMGS_PER_TEST_PROPERTY_SET, PROPERTY_DIMS_Z))

        infer.synthesize(restructured_generator,
                         num=IMGS_PER_TEST_PROPERTY_SET,
                         save_dir=img_save_dir,
                         z=z,
                         label=label_,
                         img_name_start_idx=current_idx,
                         verbose=False)

        for image_no in range(current_idx, current_idx + IMGS_PER_TEST_PROPERTY_SET):
            eyes_labels.append(label[0])
            mouth_labels.append(label[1])
            pose_labels.append(label[2])

            image_path = f'{img_save_dir}/{image_no:06d}.jpg'
            image = read_image(image_path)
            image = stylegan_transform(image)

            with torch.no_grad():
                property_scores = property_score_cnn(image.unsqueeze(0).cuda())
                property_score_np = property_scores.detach().cpu().numpy()

                eyes_cnn_scores.append(property_score_np[0][0])
                mouth_cnn_scores.append(property_score_np[0][3])
                pose_cnn_scores.append(property_score_np[0][4])

        current_idx += IMGS_PER_TEST_PROPERTY_SET

    # save label & Property CNN-derived score as csv file
    label_count = len(labels) * IMGS_PER_TEST_PROPERTY_SET

    label_info_dict = {
        'image_no': list(range(label_count)),
        'eyes_label': list(np.round(eyes_labels, 4)),
        'mouth_label': list(np.round(mouth_labels, 4)),
        'pose_label': list(np.round(pose_labels, 4)),
        'eyes_cnn_score': list(np.round(eyes_cnn_scores, 4)),
        'mouth_cnn_score': list(np.round(mouth_cnn_scores, 4)),
        'pose_cnn_score': list(np.round(pose_cnn_scores, 4))
    }
    label_info_df = pd.DataFrame(label_info_dict)
    label_info_df.to_csv(f'{img_save_dir}/label_info.csv', index=False)

    # compute corr-coef & MAE and return
    eyes_corr = np.corrcoef(eyes_labels, eyes_cnn_scores)[0][1]
    mouth_corr = np.corrcoef(mouth_labels, mouth_cnn_scores)[0][1]
    pose_corr = np.corrcoef(pose_labels, pose_cnn_scores)[0][1]

    eyes_mae = sum(abs(eyes_labels[i] - eyes_cnn_scores[i]) for i in range(label_count)) / label_count
    mouth_mae = sum(abs(mouth_labels[i] - mouth_cnn_scores[i]) for i in range(label_count)) / label_count
    pose_mae = sum(abs(pose_labels[i] - pose_cnn_scores[i]) for i in range(label_count)) / label_count

    return {'eyes_corr': eyes_corr, 'mouth_corr': mouth_corr, 'pose_corr': pose_corr,
            'eyes_mae': eyes_mae, 'mouth_mae': mouth_mae, 'pose_mae': pose_mae}


# StyleGAN Fine Tuning 에서 Discriminator 테스트용으로 real, fake 이미지 저장
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - reals (Tensor) : Real Images
# - fakes (Tensor) : Fake Images

def save_real_fake_imgs(fakes):
    image_save_dir_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/inference_test_real_fake'
    os.makedirs(image_save_dir_path, exist_ok=True)

    max_val, min_val = 1.0, -1.0

    for idx, image in enumerate(fakes):
        img_ = np.array(image.detach().cpu())
        img_ = np.transpose(img_, (1, 2, 0))
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_ = (img_ - min_val) * 255 / (max_val - min_val)

        result, overlay_image_arr = cv2.imencode(ext='.png',
                                                 img=img_,
                                                 params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

        if result:
            image_save_path = f'{image_save_dir_path}/fake_{idx:06d}.png'
            with open(image_save_path, mode='w+b') as f:
                overlay_image_arr.tofile(f)


# 모델 Fine Tuning 실시
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - restructured_generator     (nn.Module)  : StyleGAN 모델의 새로운 구조의 Generator
# - restructured_discriminator (nn.Module)  : StyleGAN 모델의 새로운 구조의 Discriminator
# - stylegan_ft_loader         (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator     (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Generator
# - fine_tuned_discriminator (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Discriminator

def run_fine_tuning(restructured_generator, restructured_discriminator, stylegan_ft_loader):

    gen_train_args = dict(w_moving_decay=0.995, style_mixing_prob=0.0,
                          trunc_psi=1.0, trunc_layers=0, randomize_noise=True)

    # run Fine-Tuning
    train(restructured_generator, restructured_discriminator, stylegan_ft_loader, gen_train_args)
