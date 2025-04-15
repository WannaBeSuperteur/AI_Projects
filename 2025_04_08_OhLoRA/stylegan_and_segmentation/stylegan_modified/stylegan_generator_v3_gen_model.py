
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph

from stylegan_modified.fine_tuning import concatenate_property_scores
from stylegan_modified.stylegan_generator import StyleGANGeneratorForV2

import numpy as np
import os
import sys

global_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
sys.path.append(global_path)

from global_common.visualize_tensor import save_tensor_png


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/model_structure_pdf'

TENSOR_VISUALIZE_TEST_BATCH_SIZE = 30
IMGS_PER_TEST_PROPERTY_SET = 10

TRAIN_BATCH_SIZE = 16
EARLY_STOPPING_ROUNDS = 10
STEP_GROUP_SIZE = 50

IMG_RES = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 7           # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

CNN_TENSOR_TEST_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/tensor_visualize_test_cnn'
os.makedirs(CNN_TENSOR_TEST_DIR, exist_ok=True)


# Loss Function for VAE

def vae_loss_function(x_reconstructed, x, mu, logvar):
    mse_loss = F.mse_loss(x, x_reconstructed, reduction='mean')
    kl_divergence_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())

    loss_dict = {'mse': mse_loss, 'kld': kl_divergence_loss}
    return mse_loss + kl_divergence_loss, loss_dict


# Conditional-VAE Encoder for StyleGAN-FineTune-v3

class CVAEEncoder(nn.Module):
    def __init__(self):
        super(CVAEEncoder, self).__init__()

        # Conv Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )

        # Fully-Connected Layer
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 8 + PROPERTY_DIMS_Z, 512),
            nn.ELU()
        )
        self.fc2_mu = nn.Linear(512 + PROPERTY_DIMS_Z, 512)
        self.fc2_var = nn.Linear(512 + PROPERTY_DIMS_Z, 512)

    def forward(self, x, property_label):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(-1, 256 * 8 * 8)
        x = torch.concat([x, property_label], dim=1)
        x = self.fc1(x)
        x = torch.concat([x, property_label], dim=1)

        z_mu = self.fc2_mu(x)
        z_var = self.fc2_var(x)

        return z_mu, z_var


# Entire model of StyleGAN-FineTune-v3

class StyleGANFineTuneV3(nn.Module):
    def __init__(self):
        super(StyleGANFineTuneV3, self).__init__()

        self.CVAE_encoder = CVAEEncoder()
        self.stylegan_generator = StyleGANGeneratorForV2(resolution=IMG_RES)

        kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
        self.stylegan_generator.G_kwargs_val = kwargs_val

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, property_label):
        mu, logvar = self.CVAE_encoder(x, property_label)
        z = self.reparameterize(mu, logvar)
        generated_image = self.stylegan_generator(z, property_label, style_mixing_prob=0.0)

        return generated_image['image'], mu, logvar


# StyleGAN-FineTune-v3 모델 정의 및 generator 의 state_dict 를 로딩
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - device    (device)    : 모델을 mapping 시킬 device (GPU 등)
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)

# Returns:
# - stylegan_finetune_v3 (nn.Module) : 학습할 StyleGAN-FineTune-v3 모델

def define_stylegan_finetune_v3(device, generator):
    stylegan_finetune_v3 = StyleGANFineTuneV3()
    stylegan_finetune_v3.optimizer = torch.optim.AdamW(stylegan_finetune_v3.parameters(), lr=0.00005)
    stylegan_finetune_v3.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=stylegan_finetune_v3.optimizer,
                                                                                T_max=10,
                                                                                eta_min=0)

    stylegan_finetune_v3.to(device)
    stylegan_finetune_v3.device = device

    # load state dict of generator
    stylegan_finetune_v3.stylegan_generator.load_state_dict(generator.state_dict())

    # save model graph of StyleGAN-FineTune-v3 before training
    model_graph = draw_graph(stylegan_finetune_v3,
                             input_data=[torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, 3, IMG_RES, IMG_RES)),
                                         torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, PROPERTY_DIMS_Z))],
                             depth=5)

    visual_graph = model_graph.visual_graph

    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/stylegan_finetune_v3.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    return stylegan_finetune_v3


# 정의된 StyleGAN-FineTune-v3 모델의 Layer 를 Freeze 처리 (CNN은 모두, Generator 는 Dense Layer 제외 모두)
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - stylegan_finetune_v3 (nn.Module) : 학습할 StyleGAN-FineTune-v3 모델
# - check_again          (bool)      : freeze 여부 재 확인 테스트용

def freeze_stylegan_finetune_v3_layers(stylegan_finetune_v3, check_again=False):

    # StyleGAN-FineTune-v3 freeze 범위 : Z -> W mapping 을 제외한 모든 레이어
    for name, param in stylegan_finetune_v3.named_parameters():
        if name.split('.')[1] != 'mapping':
            param.requires_grad = False

    # 제대로 freeze 되었는지 확인
    if check_again:
        for idx, param in enumerate(stylegan_finetune_v3.parameters()):
            print(f'StyleGAN-FineTune-v3 layer {idx} : {param.requires_grad}')


# 정의된 StyleGAN-FineTune-v3 모델을 학습
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - stylegan_finetune_v3   (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator
#                                      (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)

def run_training_stylegan_finetune_v3(stylegan_finetune_v3, fine_tuning_dataloader):
    stylegan_finetune_v3.train()

    current_epoch = 0
    min_train_loss = None
    min_train_loss_epoch = 0
    best_epoch_model = None

    while True:
        train_loss = 0.0
        train_loss_mse = 0.0
        train_loss_kld = 0.0
        total = 0

        for idx, raw_data in enumerate(fine_tuning_dataloader):
            images = raw_data['image']
            images = images.to(stylegan_finetune_v3.device)

            labels = concatenate_property_scores(raw_data)
            labels = labels.to(stylegan_finetune_v3.device).to(torch.float32)

            reconstructed_images, mu, logvar = stylegan_finetune_v3(x=images, property_label=labels)
            stylegan_finetune_v3.optimizer.zero_grad()

            loss, loss_dict = vae_loss_function(reconstructed_images, images, mu, logvar)
            loss.backward()

            train_loss += float(loss.detach().cpu().numpy())
            train_loss_mse += float(loss_dict['mse'].detach().cpu().numpy())
            train_loss_kld += float(loss_dict['kld'].detach().cpu().numpy())

            total += labels.size(0)

            stylegan_finetune_v3.optimizer.step()

        train_loss /= total
        train_loss_mse /= total
        train_loss_kld /= total

        print(f'epoch {current_epoch}: loss = {train_loss:.4f} (mse: {train_loss_mse:.4f}, kld: {train_loss_kld:.4f})')

        current_epoch += 1
        stylegan_finetune_v3.scheduler.step()

        # Early Stopping 처리
        if min_train_loss is None or train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_loss_epoch = current_epoch

            best_epoch_model = StyleGANFineTuneV3().to(stylegan_finetune_v3.device)
            best_epoch_model.load_state_dict(stylegan_finetune_v3.state_dict())

        if current_epoch - min_train_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        # 이미지 생성 테스트
        test_create_output_images(stylegan_finetune_v3, current_epoch)

    fine_tuned_generator = best_epoch_model.stylegan_generator
    return fine_tuned_generator


# StyleGAN-FineTune-v3 모델 학습 중 출력 결과물 테스트
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - stylegan_finetune_v3 (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (StyleGAN-FineTune-v3 으로 Fine-Tuning 중)
# - current_epoch        (int)       : 현재 epoch 의 번호

def test_create_output_images(stylegan_finetune_v3, current_epoch):
    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_during_finetuning_v3'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}'
    os.makedirs(img_save_dir, exist_ok=True)

    # label: 'eyes', 'hair_color', 'hair_length', 'mouth', 'pose', 'background_mean' (, 'background_std')
    z = torch.randn((IMGS_PER_TEST_PROPERTY_SET, ORIGINAL_HIDDEN_DIMS_Z)).to(torch.float32)

    labels = [[ 1.2,  1.2,  1.2, -1.2, -1.2,  1.2, 0.0],
              [-1.2,  1.2,  1.2, -1.2, -1.2,  1.2, 0.0],
              [-1.2, -1.2,  1.2, -1.2, -1.2,  1.2, 0.0],
              [-1.2, -1.2, -1.2, -1.2, -1.2,  1.2, 0.0],
              [-1.2, -1.2, -1.2,  1.2, -1.2,  1.2, 0.0],
              [-1.2, -1.2, -1.2,  1.2,  1.2,  1.2, 0.0],
              [-1.2, -1.2, -1.2,  1.2,  1.2, -1.2, 0.0]]

    for label_idx, label in enumerate(labels):
        label_np = np.array([IMGS_PER_TEST_PROPERTY_SET * [label]])
        label_np = label_np.reshape((IMGS_PER_TEST_PROPERTY_SET, PROPERTY_DIMS_Z))
        label_torch = torch.tensor(label_np).to(torch.float32)

        with torch.no_grad():
            generated_images = stylegan_finetune_v3.stylegan_generator(z=z.cuda(), label=label_torch.cuda())['image']
            generated_images = generated_images.detach().cpu()
        image_count = generated_images.size(0)

        for img_idx in range(image_count):
            img_no = label_idx * IMGS_PER_TEST_PROPERTY_SET + img_idx

            save_tensor_png(generated_images[img_idx],
                            image_save_path=f'{img_save_dir}/test_img_{img_no}.png')


# StyleGAN-FineTune-v3 모델 학습
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - device                 (device)     : 모델을 mapping 시킬 device (GPU 등)
# - generator              (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator
#                                      (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)

def train_stylegan_finetune_v3(device, generator, fine_tuning_dataloader):

    # define StyleGAN-FineTune-v3 model
    stylegan_finetune_v3 = define_stylegan_finetune_v3(device, generator)
    freeze_stylegan_finetune_v3_layers(stylegan_finetune_v3)

    # run Fine-Tuning
    fine_tuned_generator = run_training_stylegan_finetune_v3(stylegan_finetune_v3, fine_tuning_dataloader)

    return fine_tuned_generator