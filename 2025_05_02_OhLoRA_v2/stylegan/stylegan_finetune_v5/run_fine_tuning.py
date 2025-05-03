# Modified Loss implementation from https://github.com/genforce/genforce/blob/master/runners/losses/logistic_gan_loss.py
# Modified Train Process implementation from https://github.com/genforce/genforce/blob/master/runners/stylegan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_gan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_runner.py
# Train Argument Settings from https://github.com/genforce/genforce/blob/master/configs/stylegan_demo.py


import os
import time
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import cv2
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import stylegan_common.stylegan_generator_inference as infer


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


def compute_d_loss(generator, discriminator, data, gen_train_args, dis_train_args, r1_gamma, r2_gamma, save_image):
    """Computes loss for discriminator."""

    reals = data['image']
    labels = data['label']
    reals.requires_grad = True

    latents = torch.randn(reals.shape[0], ORIGINAL_HIDDEN_DIMS_Z).cuda()
    latents.requires_grad = True
    # TODO: Use random labels.
    fakes = generator(latents, label=labels, **gen_train_args)['image']

    if save_image:
        save_real_fake_imgs(reals, fakes)

    real_scores = discriminator(reals, label=labels, **dis_train_args)
    fake_scores = discriminator(fakes, label=labels, **dis_train_args)

    d_loss = F.softplus(fake_scores).mean()
    d_loss += F.softplus(-real_scores).mean()

    real_grad_penalty = torch.zeros_like(d_loss)
    fake_grad_penalty = torch.zeros_like(d_loss)
    if r1_gamma:
        real_grad_penalty = compute_grad_penalty(reals, real_scores)
    if r2_gamma:
        fake_grad_penalty = compute_grad_penalty(fakes, fake_scores)

    real_scores_np = real_scores.detach().cpu().numpy()
    fake_scores_np = fake_scores.detach().cpu().numpy()

    real_scores_mean = np.mean(real_scores_np.flatten())
    fake_scores_mean = np.mean(fake_scores_np.flatten())

    all_scores = np.concatenate([real_scores_np, fake_scores_np])
    all_labels = np.concatenate([np.ones(len(real_scores_np)), np.zeros(len(fake_scores_np))])
    real_fake_auroc = roc_auc_score(all_labels, all_scores)

    return (d_loss +
            real_grad_penalty * (r1_gamma * 0.5) +
            fake_grad_penalty * (r2_gamma * 0.5)), real_scores_mean, fake_scores_mean, real_fake_auroc


def compute_g_loss(generator, discriminator, data, gen_train_args, dis_train_args):  # pylint: disable=no-self-use
    """Computes loss for generator."""
    # TODO: Use random labels.

    batch_size = data['image'].shape[0]
    labels = data['label']

    latents = torch.randn(batch_size, ORIGINAL_HIDDEN_DIMS_Z).cuda()
    fakes = generator(latents, label=labels, **gen_train_args)['image']
    fake_scores = discriminator(fakes, label=labels, **dis_train_args)

    g_loss = F.softplus(-fake_scores).mean()
    return g_loss


# generator     -> layers_to_train = ['mapping']
# discriminator -> layers_to_train = ['layer12', 'layer13', 'layer14']
def set_model_requires_grad(model, model_name, requires_grad):
    """Sets the `requires_grad` configuration for a particular model."""

    assert model_name in ['generator', 'discriminator']

    for name, param in model.named_parameters():

        if requires_grad:  # requires_grad == True
            if model_name == 'generator':
                if name.split('.')[0] == 'mapping':
                    param.requires_grad = True

            elif model_name == 'discriminator':
                if name.split('.')[0] in ['layer12', 'layer13', 'layer14']:
                    param.requires_grad = True

        else:
            param.requires_grad = False


def moving_average_model(model, avg_model, beta=0.999):
    """Moving average model weights.

    This trick is commonly used in GAN training, where the weight of the
    generator is life-long averaged

    Args:
        model: The latest model used to update the averaged weights.
        avg_model: The averaged model weights.
        beta: Hyper-parameter used for moving average.
    """
    model_params = dict(model.named_parameters())
    avg_params = dict(avg_model.named_parameters())

    assert len(model_params) == len(avg_params)
    for param_name in avg_params:
        assert param_name in model_params
        avg_params[param_name].data = (
                avg_params[param_name].data * beta +
                model_params[param_name].data * (1 - beta))


def train_step(generator, generator_smooth, discriminator, data, gen_train_args, dis_train_args,
               r1_gamma, r2_gamma, g_smooth_img, save_image):

    # Update discriminator.
    set_model_requires_grad(discriminator, 'discriminator', True)
    set_model_requires_grad(generator, 'generator', False)
#    check_model_trainable_status(0, generator, discriminator)

    d_loss, real_scores_mean, fake_scores_mean, real_fake_auroc = compute_d_loss(generator, discriminator, data,
                                                                                 gen_train_args, dis_train_args,
                                                                                 r1_gamma, r2_gamma, save_image)

    discriminator.optimizer.zero_grad()
    d_loss.backward()
    discriminator.optimizer.step()
    d_loss_float = float(d_loss.detach().cpu())

    # Update generator.
    set_model_requires_grad(discriminator, 'discriminator', False)
    set_model_requires_grad(generator, 'generator', True)
#    check_model_trainable_status(1, generator, discriminator)

    g_train_count = 0
    g_loss_float = None

    while g_train_count < 4:
        g_loss = compute_g_loss(generator, discriminator, data, gen_train_args, dis_train_args)
        generator.optimizer.zero_grad()
        g_loss.backward()
        generator.optimizer.step()

        g_loss_float = float(g_loss.detach().cpu())
        g_train_count += 1

        if g_loss_float < 2.0 * d_loss_float:
            break

    return d_loss_float, g_loss_float, g_train_count, real_scores_mean, fake_scores_mean, real_fake_auroc


def train(generator, generator_smooth, discriminator, stylegan_ft_loader, gen_train_args, dis_train_args,
          r1_gamma, r2_gamma, g_smooth_img):

    """Training function."""
    print('Start training.')

    train_log_dict = {'epoch': [], 'idx': [], 'd_loss': [], 'g_loss': [], 'g_train_count': [],
                      'real_scores_mean': [], 'fake_scores_mean': [], 'real_fake_auroc': []}

    current_epoch = 0

    gen_save_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/stylegan_gen_fine_tuned_v5.pth'
    dis_save_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/stylegan_dis_fine_tuned_v5.pth'
    train_log_save_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/train_log.csv'

    while current_epoch < TOTAL_EPOCHS:
        for idx, raw_data in enumerate(stylegan_ft_loader):
            concatenated_labels = concatenate_property_scores(raw_data)

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

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/inference_test_during_finetuning'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}_idx_{batch_idx:04d}'

    # label: 'eyes', 'mouth', 'pose'
    current_idx = 0

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

        current_idx += IMGS_PER_TEST_PROPERTY_SET


# StyleGAN Fine Tuning 에서 Discriminator 테스트용으로 real, fake 이미지 저장
# Create Date : 2025.05.03
# Last Update Date : -

# Arguments:
# - reals (Tensor) : Real Images
# - fakes (Tensor) : Fake Images

def save_real_fake_imgs(reals, fakes):
    image_lists = [reals, fakes]
    real_fake_label = ['real', 'fake']
    image_save_dir_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/inference_test_real_fake'
    os.makedirs(image_save_dir_path, exist_ok=True)

    max_val, min_val = 1.0, -1.0

    for image_list, real_fake in zip(image_lists, real_fake_label):
        for idx, image in enumerate(image_list):
            img_ = np.array(image.detach().cpu())
            img_ = np.transpose(img_, (1, 2, 0))
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_ = (img_ - min_val) * 255 / (max_val - min_val)

            result, overlay_image_arr = cv2.imencode(ext='.png',
                                                     img=img_,
                                                     params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

            if result:
                image_save_path = f'{image_save_dir_path}/{real_fake}_{idx:06d}.png'
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