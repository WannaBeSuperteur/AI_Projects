# Modified Loss implementation from https://github.com/genforce/genforce/blob/master/runners/losses/logistic_gan_loss.py
# Modified Train Process implementation from https://github.com/genforce/genforce/blob/master/runners/stylegan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_gan_runner.py
#                                            https://github.com/genforce/genforce/blob/master/runners/base_runner.py
# Train Argument Settings from https://github.com/genforce/genforce/blob/master/configs/stylegan_demo.py


import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import read_image

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import stylegan_common.stylegan_generator_inference as modified_inf
from generate_dataset.cnn_common import load_cnn_model
from generate_dataset.cnn_gender import GenderCNN
from generate_dataset.cnn_quality import QualityCNN
from generate_dataset.cnn_age import AgeCNN
from generate_dataset.cnn_glass import GlassCNN


ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS = 7
TRAIN_BATCH_SIZE = 8
TOTAL_EPOCHS = 500
IMGS_PER_TEST_PROPERTY_SET = 100
CNN_MODELS_FOR_EACH_OF_GENDER_QUALITY_AGE_GLASS = 5

gender_cnn_models = load_cnn_model(property_name='gender', cnn_model_class=GenderCNN)
quality_cnn_models = load_cnn_model(property_name='quality', cnn_model_class=QualityCNN)
age_cnn_models = load_cnn_model(property_name='age', cnn_model_class=AgeCNN)
glass_cnn_models = load_cnn_model(property_name='glass', cnn_model_class=GlassCNN)

base_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])


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


# generator     -> Z -> W mapping & synthesis layer 의 4 x 4 output 까지 학습 가능
#                  layers_to_train = ['mapping', 'synthesis.layer0', 'synthesis.layer1', 'synthesis.output0']

# discriminator -> 대부분의 Conv. Layer 를 Freeze
#                  layers_to_train = ['layer12', 'layer13', 'layer14']

def set_model_requires_grad(model, model_name, requires_grad):
    """Sets the `requires_grad` configuration for a particular model."""

    assert model_name in ['generator', 'discriminator']

    for name, param in model.named_parameters():

        if requires_grad:  # requires_grad == True
            if model_name == 'generator':
                trainable_synthesis_layers = ['layer0', 'layer1', 'output0']

                if name.split('.')[0] == 'mapping':
                    param.requires_grad = True
                elif name.split('.')[0] == 'synthesis' and name.split('.')[1] in trainable_synthesis_layers:
                    param.requires_grad = True

            elif model_name == 'discriminator':
                if name.split('.')[0] in ['layer12', 'layer13', 'layer14']:
                    param.requires_grad = True

        else:
            param.requires_grad = False


def train_step(generator, discriminator, data, gen_train_args, dis_train_args, r1_gamma, r2_gamma, save_image):

    # Update discriminator.
    set_model_requires_grad(discriminator, 'discriminator', True)
    set_model_requires_grad(generator, 'generator', False)
#    check_model_trainable_status(0, generator, discriminator) (GEN/DIS layer frozen/trainable 상태 정상 확인 완료)

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
#    check_model_trainable_status(1, generator, discriminator) (GEN/DIS layer frozen/trainable 상태 정상 확인 완료)

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


def train(generator, discriminator, stylegan_ft_loader, gen_train_args, dis_train_args, r1_gamma, r2_gamma):

    """Training function."""
    print('Start training.')

    train_log_dict = {'epoch': [], 'idx': [], 'd_loss': [], 'g_loss': [], 'g_train_count': [],
                      'real_scores_mean': [], 'fake_scores_mean': [], 'real_fake_auroc': [],
                      'mean_gender_score': [], 'mean_quality_score': [], 'mean_age_score': [], 'mean_glass_score': []}

    current_epoch = 0

    gen_save_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v8.pth'
    dis_save_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_dis_fine_tuned_v8.pth'
    train_log_save_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v8/train_log.csv'

    while current_epoch < TOTAL_EPOCHS:
        for idx, raw_data in enumerate(stylegan_ft_loader):
            current_batch_size = len(raw_data['image'])
            labels = torch.zeros((current_batch_size, ORIGINALLY_PROPERTY_DIMS))
            labels = labels.to(torch.float32)

            data = {
                'image': raw_data['image'].cuda(),
                'label': labels.cuda()
            }

            print_result_and_save_image = (idx % 50 == 0 or
                                           (current_epoch == 0 and idx % 25 == 0) or
                                           (current_epoch == 0 and idx < 10))

            d_loss_float, g_loss_float, g_train_count, real_scores_mean, fake_scores_mean, real_fake_auroc =(
                train_step(generator, discriminator, data, gen_train_args, dis_train_args, r1_gamma, r2_gamma,
                           save_image=print_result_and_save_image))

            if print_result_and_save_image:
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

                score_mean_dict = log_gender_quality_age_glass_predicted_score(current_epoch,
                                                                               batch_idx=idx,
                                                                               train_log_dict=train_log_dict)

                print(f'epoch={current_epoch}, idx={idx}, '
                      f'd_loss={d_loss_float:.4f}, g_loss={g_loss_float:.4f}, g_train_count={g_train_count}, '
                      f'real_scores_mean={real_scores_mean:.4f}, fake_scores_mean={fake_scores_mean:.4f}, '
                      f'real_fake_auroc={real_fake_auroc:.4f}, inference_result={score_mean_dict}')

                pd.DataFrame(train_log_dict).to_csv(train_log_save_path)

        # save model for EVERY EPOCH
        torch.save(generator.state_dict(), gen_save_path)
        torch.save(discriminator.state_dict(), dis_save_path)

        current_epoch += 1


# 모델의 각 레이어의 trainable / fronzen 상태 확인
# Create Date : 2025.05.28
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
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator
# - current_epoch         (int)       : Fine-Tuning 중 현재 epoch 번호
# - batch_idx             (int)       : Fine-Tuning 중 현재 epoch 에서의 batch index 번호

# Returns:
# - stylegan_modified/inference_test_during_finetuning 에 생성 결과 저장

def run_inference_test_during_finetuning(finetune_v1_generator, current_epoch, batch_idx):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    finetune_v1_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v8/inference_test_during_finetuning'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}_idx_{batch_idx:04d}'
    os.makedirs(img_save_dir, exist_ok=True)

    z = np.random.normal(0, 1, size=(IMGS_PER_TEST_PROPERTY_SET, ORIGINAL_HIDDEN_DIMS_Z))
    label_like = np.random.normal(0, 1, size=(IMGS_PER_TEST_PROPERTY_SET, ORIGINALLY_PROPERTY_DIMS))

    modified_inf.synthesize(finetune_v1_generator,
                            num=IMGS_PER_TEST_PROPERTY_SET,
                            save_dir=img_save_dir,
                            z=z,
                            label=label_like,
                            img_name_start_idx=0,
                            verbose=False)


# Inference Test 에서 Property Score CNN 의 gender & quality & age & glass 속성 예측값 로깅
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - current_epoch  (int)  : Fine-Tuning 중 현재 epoch 번호
# - batch_idx      (int)  : Fine-Tuning 중 현재 epoch 에서의 batch index 번호
# - train_log_dict (dict) : 전체 학습 과정 로깅용 dict

# Returns:
# - score_mean_dict (dict) : Inference 대상 이미지에 대한, gender, quality, age, glass score 의 평균값 dict

def log_gender_quality_age_glass_predicted_score(current_epoch, batch_idx, train_log_dict):
    global gender_cnn_models, quality_cnn_models, age_cnn_models, glass_cnn_models

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v8/inference_test_during_finetuning'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}_idx_{batch_idx:04d}'

    # prepare inference test result dict
    inference_test_result = {
        'img_no': list(range(IMGS_PER_TEST_PROPERTY_SET)),
        'gender_score': [], 'quality_score': [], 'age_score': [], 'glass_score': []
    }
    for cnn_no in range(CNN_MODELS_FOR_EACH_OF_GENDER_QUALITY_AGE_GLASS):
        inference_test_result[f'gender_score_cnn_{cnn_no}'] = []
        inference_test_result[f'quality_score_cnn_{cnn_no}'] = []
        inference_test_result[f'age_score_cnn_{cnn_no}'] = []
        inference_test_result[f'glass_score_cnn_{cnn_no}'] = []

    # run inference test
    with torch.no_grad():
        for img_no in range(IMGS_PER_TEST_PROPERTY_SET):
            image_path = f'{img_save_dir}/{img_no:06d}.jpg'
            image = read_image(image_path)
            image = base_transform(image)
            image = image.unsqueeze(0)

            for cnn_no in range(CNN_MODELS_FOR_EACH_OF_GENDER_QUALITY_AGE_GLASS):
                output_score_gender = gender_cnn_models[cnn_no](image.cuda()).to(torch.float32).detach().cpu().numpy()
                output_score_quality = quality_cnn_models[cnn_no](image.cuda()).to(torch.float32).detach().cpu().numpy()
                output_score_age = age_cnn_models[cnn_no](image.cuda()).to(torch.float32).detach().cpu().numpy()
                output_score_glass = glass_cnn_models[cnn_no](image.cuda()).to(torch.float32).detach().cpu().numpy()

                inference_test_result[f'gender_score_cnn_{cnn_no}'].append(round(output_score_gender[0][0], 4))
                inference_test_result[f'quality_score_cnn_{cnn_no}'].append(round(output_score_quality[0][0], 4))
                inference_test_result[f'age_score_cnn_{cnn_no}'].append(round(output_score_age[0][0], 4))
                inference_test_result[f'glass_score_cnn_{cnn_no}'].append(round(output_score_glass[0][0], 4))

            for property_name in ['gender', 'quality', 'age', 'glass']:
                property_score_sum = 0.0
                for cnn_no in range(CNN_MODELS_FOR_EACH_OF_GENDER_QUALITY_AGE_GLASS):
                    property_score_sum += inference_test_result[f'{property_name}_score_cnn_{cnn_no}'][-1]
                property_score_sum /= CNN_MODELS_FOR_EACH_OF_GENDER_QUALITY_AGE_GLASS

                inference_test_result[f'{property_name}_score'].append(round(property_score_sum, 4))

    # save inference test result
    inference_test_result_path = f'{img_save_dir}/inference_test_result.csv'
    inference_test_df = pd.DataFrame(inference_test_result)
    inference_test_df.to_csv(inference_test_result_path)

    # update train log dict
    score_mean_dict = {}

    for property_name in ['gender', 'quality', 'age', 'glass']:
        score_mean = inference_test_df[f'{property_name}_score'].mean()
        train_log_dict[f'mean_{property_name}_score'].append(round(score_mean, 4))
        score_mean_dict[property_name] = round(score_mean, 4)

    return score_mean_dict


# StyleGAN Fine Tuning 에서 Discriminator 테스트용으로 real, fake 이미지 저장
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - reals (Tensor) : Real Images
# - fakes (Tensor) : Fake Images

def save_real_fake_imgs(reals, fakes):
    image_lists = [reals, fakes]
    real_fake_label = ['real', 'fake']
    image_save_dir_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v8/inference_test_real_fake'
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
# Create Date : 2025.05.28
# Last Update Date : -

# Arguments:
# - finetune_v1_generator     (nn.Module)  : StyleGAN-FineTune-v1 의 Generator
# - finetune_v1_discriminator (nn.Module)  : StyleGAN-FineTune-v1 의 Discriminator
# - stylegan_ft_loader        (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_v8_generator     (nn.Module) : Fine-Tuning 된 StyleGAN 모델 (= StyleGAN-FineTune-v8) 의 Generator
# - fine_tuned_v8_discriminator (nn.Module) : Fine-Tuning 된 StyleGAN 모델 (= StyleGAN-FineTune-v8) 의 Discriminator

def run_fine_tuning(finetune_v1_generator, finetune_v1_discriminator, stylegan_ft_loader):

    gen_train_args = dict(w_moving_decay=0.995, style_mixing_prob=0.0,
                          trunc_psi=1.0, trunc_layers=0, randomize_noise=True)
    dis_train_args = dict()

    r1_gamma = 10.0
    r2_gamma = 0.0

    # run Fine-Tuning
    train(finetune_v1_generator, finetune_v1_discriminator, stylegan_ft_loader, gen_train_args, dis_train_args,
          r1_gamma, r2_gamma)
