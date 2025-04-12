# Modified Loss implementation from https://github.com/genforce/genforce/blob/master/runners/losses/logistic_gan_loss.py
# Modified Train Process implementation from https://github.com/genforce/genforce/blob/master/runners/stylegan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_gan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_runner.py
# Train Argument Settings from https://github.com/genforce/genforce/blob/master/configs/stylegan_demo.py


import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

import stylegan_modified.stylegan_generator_inference as modified_inf


ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 5           # eyes, hair_color, hair_length, mouth, pose
TRAIN_BATCH_SIZE = 16
TOTAL_EPOCHS = 500
IMGS_PER_TEST_PROPERTY_SET = 5


def compute_grad_penalty(images, scores):
    """Computes gradient penalty."""
    image_grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=images,
        create_graph=True,
        retain_graph=True)[0].view(images.shape[0], -1)
    penalty = image_grad.pow(2).sum(dim=1).mean()
    return penalty


def compute_d_loss(generator, discriminator, data, gen_train_args, dis_train_args, r1_gamma, r2_gamma):
    """Computes loss for discriminator."""

    reals = data['image']
    labels = data['label']
    reals.requires_grad = True

    latents = torch.randn(reals.shape[0], ORIGINAL_HIDDEN_DIMS_Z).cuda()
    latents.requires_grad = True
    # TODO: Use random labels.
    fakes = generator(latents, label=labels, **gen_train_args)['image']
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

    return (d_loss +
            real_grad_penalty * (r1_gamma * 0.5) +
            fake_grad_penalty * (r2_gamma * 0.5))


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


# generator     -> params =  44 layers, named_params =  44 layers, layers_to_train = ['mapping']
# discriminator -> params = 101 layers, named_params = 101 layers, layers_to_train = ['layer12', 'layer13', 'layer14']
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
               r1_gamma, r2_gamma, g_smooth_img):

    # Update discriminator.
    set_model_requires_grad(discriminator, 'discriminator', True)
    set_model_requires_grad(generator, 'generator', False)
#    check_model_trainable_status(0, generator, discriminator)

    d_loss = compute_d_loss(generator, discriminator, data, gen_train_args, dis_train_args, r1_gamma, r2_gamma)
    discriminator.optimizer.zero_grad()
    d_loss.backward()
    discriminator.optimizer.step()

    # Life-long update for generator.
    beta = 0.5 ** (TRAIN_BATCH_SIZE / g_smooth_img)
    moving_average_model(model=generator, avg_model=generator_smooth, beta=beta)

    # Update generator.
    set_model_requires_grad(discriminator, 'discriminator', False)
    set_model_requires_grad(generator, 'generator', True)
#    check_model_trainable_status(1, generator, discriminator)

    g_loss = compute_g_loss(generator, discriminator, data, gen_train_args, dis_train_args)
    generator.optimizer.zero_grad()
    g_loss.backward()
    generator.optimizer.step()

    d_loss_float = float(d_loss.detach().cpu())
    g_loss_float = float(g_loss.detach().cpu())

    return d_loss_float, g_loss_float


def train(generator, generator_smooth, discriminator, stylegan_ft_loader, gen_train_args, dis_train_args,
          r1_gamma, r2_gamma, g_smooth_img):

    """Training function."""
    print('Start training.')

    current_epoch = 0

    while current_epoch < TOTAL_EPOCHS:
        for idx, raw_data in enumerate(stylegan_ft_loader):
            concatenated_labels = torch.concat([raw_data['label']['eyes'],
                                                raw_data['label']['hair_color'],
                                                raw_data['label']['hair_length'],
                                                raw_data['label']['mouth'],
                                                raw_data['label']['pose']])
            concatenated_labels = torch.reshape(concatenated_labels, (PROPERTY_DIMS_Z, -1))
            concatenated_labels = torch.transpose(concatenated_labels, 0, 1)
            concatenated_labels = concatenated_labels.to(torch.float32)

            data = {
                'image': raw_data['image'].cuda(),
                'label': concatenated_labels.cuda()
            }

            d_loss_float, g_loss_float = train_step(generator, generator_smooth, discriminator, data,
                                                    gen_train_args, dis_train_args, r1_gamma, r2_gamma, g_smooth_img)

            if idx % 10 == 0 or (current_epoch == 0 and idx < 10):
                print(f'epoch={current_epoch}, idx={idx}, d_loss={d_loss_float:.4f}, g_loss={g_loss_float:.4f}')
                run_inference_test_during_finetuning(generator, current_epoch=current_epoch, batch_idx=idx)

        current_epoch += 1


# 모델의 각 레이어의 trainable / fronzen 상태 확인
# Create Date : 2025.04.12
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
# Create Date : 2025.04.12
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

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_during_finetuning'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}_idx_{batch_idx:04d}'

    # label: 'eyes', 'hair_color', 'hair_length', 'mouth', 'pose'
    current_idx = 0

    labels = [[ 1.5,  1.5,  1.2, -1.0, -1.0],
              [-1.5,  1.5,  1.2, -1.0, -1.0],
              [-1.5, -1.5,  1.2, -1.0, -1.0],
              [-1.5, -1.5, -1.8, -1.0, -1.0],
              [-1.5, -1.5, -1.8,  2.0, -1.0],
              [-1.5, -1.5, -1.8,  2.0,  2.0]]

    for label in labels:
        label_ = np.array([IMGS_PER_TEST_PROPERTY_SET * [label]])
        label_ = label_.reshape((IMGS_PER_TEST_PROPERTY_SET, PROPERTY_DIMS_Z))

        modified_inf.synthesize(restructured_generator,
                                num=IMGS_PER_TEST_PROPERTY_SET,
                                save_dir=img_save_dir,
                                z=None,
                                label=label_,
                                img_name_start_idx=current_idx,
                                verbose=False)

        current_idx += IMGS_PER_TEST_PROPERTY_SET


# 모델 Fine Tuning 실시
# Create Date : 2025.04.12
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
    g_smooth_img = 10000

    # copy Re-constructed Generator Model
    restructured_generator_smooth = deepcopy(restructured_generator)

    # run Fine-Tuning
    train(restructured_generator, restructured_generator_smooth, restructured_discriminator,
          stylegan_ft_loader, gen_train_args, dis_train_args,
          r1_gamma, r2_gamma, g_smooth_img)

    raise NotImplementedError