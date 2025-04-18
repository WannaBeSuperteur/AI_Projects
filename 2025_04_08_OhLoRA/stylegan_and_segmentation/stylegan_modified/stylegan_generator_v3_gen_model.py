
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchview import draw_graph
from torchvision.io import read_image

from stylegan_modified.fine_tuning import concatenate_property_scores
from stylegan_modified.stylegan_generator import StyleGANGeneratorForV3
from stylegan_modified.stylegan_generator_v2_cnn import PropertyScoreCNN
from cnn.cnn_gender import GenderCNN

import numpy as np
import pandas as pd
import os
import sys

global_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
sys.path.append(global_path)

from global_common.visualize_tensor import save_tensor_png


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/model_structure_pdf'

TENSOR_VISUALIZE_TEST_BATCH_SIZE = 8
IMGS_PER_TEST_PROPERTY_SET = 30
LAST_BATCHES_TO_SAVE_MU_AND_LOGVAR = 100
RANDOM_GEN_TEST_IMGS_PER_EPOCH = 30

assert IMGS_PER_TEST_PROPERTY_SET <= LAST_BATCHES_TO_SAVE_MU_AND_LOGVAR

TRAIN_BATCH_SIZE = 16
EARLY_STOPPING_ROUNDS = 40
MAX_EPOCHS = 1000

IMG_RES = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 7           # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

CNN_TENSOR_TEST_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/tensor_visualize_test_cnn'
os.makedirs(CNN_TENSOR_TEST_DIR, exist_ok=True)

torch.set_printoptions(linewidth=160, sci_mode=False)

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


loss_types = ['eyes', 'hair_color', 'hair_length', 'mouth', 'pose', 'back_mean', 'back_std']

mus = []
log_vars = []


# Loss Function for VAE (Background std 제외)

def vae_loss_function(generated_image_property_score, generated_image_gender_score, labels, mu, logvar):
    n = labels.size(0)

    mse_loss_eyes = F.mse_loss(generated_image_property_score[:, :1], labels[:, :1], reduction='mean')
    mse_loss_mouth = F.mse_loss(generated_image_property_score[:, 3:4], labels[:, 3:4], reduction='mean')
    mse_loss_pose = F.mse_loss(generated_image_property_score[:, 4:5], labels[:, 4:5], reduction='mean')
    mse_loss = (mse_loss_eyes + mse_loss_mouth + mse_loss_pose) / 3.0

    gender_loss = F.mse_loss(generated_image_gender_score, torch.ones((n, 1)).cuda(), reduction='mean')
    mu_loss = F.mse_loss(mu, torch.zeros((n, ORIGINAL_HIDDEN_DIMS_Z)).cuda(), reduction='mean')
    logvar_loss = F.mse_loss(logvar, torch.zeros((n, ORIGINAL_HIDDEN_DIMS_Z)).cuda(), reduction='mean')

    total_loss = mse_loss + gender_loss + 0.2 * mu_loss + 0.05 * logvar_loss

    loss_dict = {'total_loss': round(float(total_loss.detach().cpu().numpy()), 4),
                 'mse': round(float(mse_loss.detach().cpu().numpy()), 4),
                 'gender_loss': round(float(gender_loss.detach().cpu().numpy()), 4),
                 'mu_loss': round(float(mu_loss.detach().cpu().numpy()), 4),
                 'logvar_loss': round(float(logvar_loss.detach().cpu().numpy()), 4)}

    for idx, loss_type in enumerate(loss_types):
        loss_of_type_mse = nn.MSELoss()(generated_image_property_score[:, idx:idx+1], labels[:, idx:idx+1])
        loss_of_type_mse = float(loss_of_type_mse.detach().cpu().numpy())

        loss_of_type_abs = nn.L1Loss()(generated_image_property_score[:, idx:idx + 1], labels[:, idx:idx + 1])
        loss_of_type_abs = float(loss_of_type_abs.detach().cpu().numpy())

        loss_dict[f'{loss_type}_mse'] = round(loss_of_type_mse, 4)
        loss_dict[f'{loss_type}_abs'] = round(loss_of_type_abs, 4)

    return total_loss, loss_dict


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
        self.stylegan_generator = StyleGANGeneratorForV3(resolution=IMG_RES)
        self.property_score_cnn = PropertyScoreCNN()
        self.gender_cnn = GenderCNN()

        kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
        self.stylegan_generator.G_kwargs_val = kwargs_val

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, property_label,
                tensor_visualize_test=False,
                use_mu_and_logvar_for_test_generation=False,
                save_mu_and_logvar=False):

        global mus, log_vars

        mu, logvar = self.CVAE_encoder(x, property_label)
        z = self.reparameterize(mu, logvar)

        if use_mu_and_logvar_for_test_generation or save_mu_and_logvar:
            mus.append(list(mu[0].detach().cpu().numpy()))
            log_vars.append(list(logvar[0].detach().cpu().numpy()))

        generated_image = self.stylegan_generator(z, property_label, style_mixing_prob=0.0)
        generated_image_property_score = self.property_score_cnn(generated_image['image'])
        generated_image_gender_score = self.gender_cnn(generated_image['image'])

        if tensor_visualize_test:
            test_name = 'test_during_finetune'

            current_batch_size = generated_image['image'].size(0)
            property_score_np = generated_image_property_score.detach().cpu().numpy()
            gender_score_np = generated_image_gender_score.detach().cpu().numpy()

            property_score_info_dict = {
                'img_no': list(range(current_batch_size)),
                'gender_score': list(gender_score_np[:, 0]),
                'eyes_score': list(property_score_np[:, 0]),
                'hair_color_score': list(property_score_np[:, 1]),
                'hair_length_score': list(property_score_np[:, 2]),
                'mouth_score': list(property_score_np[:, 3]),
                'pose_score': list(property_score_np[:, 4]),
                'back_mean_score': list(property_score_np[:, 5]),
                'back_std_score': list(property_score_np[:, 6])
            }
            property_score_info_df = pd.DataFrame(property_score_info_dict)
            property_score_info_df.to_csv(f'{CNN_TENSOR_TEST_DIR}/finetune_v3_{test_name}_result.csv',
                                          index=False)

            for i in range(current_batch_size):
                save_tensor_png(generated_image['image'][i],
                                image_save_path=f'{CNN_TENSOR_TEST_DIR}/finetune_v3_{test_name}_{i:03d}.png')

        return mu, logvar, generated_image_property_score, generated_image_gender_score


# StyleGAN-FineTune-v3 모델 정의 및 generator 의 state_dict 를 로딩
# Create Date : 2025.04.15
# Last Update Date : 2025.04.18
# - CVAE Encoder 의 learning rate 를 기존의 1/10 으로 감소

# Arguments:
# - device             (device)    : 모델을 mapping 시킬 device (GPU 등)
# - generator          (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - property_cnn_model (nn.Module) : StyleGAN-FineTune-v3 Fine-Tuning 에 사용할 핵심 속성값 도출용 학습된 CNN 모델
# - gender_cnn_model   (nn.Module) : StyleGAN-FineTune-v3 Fine-Tuning 에 사용할 성별 판단용 학습된 CNN 모델

# Returns:
# - stylegan_finetune_v3 (nn.Module) : 학습할 StyleGAN-FineTune-v3 모델

def define_stylegan_finetune_v3(device, generator, property_cnn_model, gender_cnn_model):
    stylegan_finetune_v3 = StyleGANFineTuneV3()

    stylegan_finetune_v3.optimizer = torch.optim.AdamW(
        [{'params': stylegan_finetune_v3.stylegan_generator.parameters()},
         {'params': stylegan_finetune_v3.CVAE_encoder.parameters(), 'lr': 0.00001}],
        lr=0.0001)

    stylegan_finetune_v3.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=stylegan_finetune_v3.optimizer,
                                                                                T_max=10,
                                                                                eta_min=0)

    stylegan_finetune_v3.to(device)
    stylegan_finetune_v3.device = device

    # load state dict of generator
    # (size mismatch of property score mapping layers between StyleGAN-FineTune-v1,v2 and StyleGAN-FineTune-v3)
    stylegan_finetune_v3_generator_state_dict = {}

    for k in generator.state_dict().keys():
        if k not in ['mapping.label_weight', 'mapping.dense0.weight']:
            stylegan_finetune_v3_generator_state_dict[k] = generator.state_dict()[k]

    stylegan_finetune_v3.stylegan_generator.load_state_dict(stylegan_finetune_v3_generator_state_dict, strict=False)

    # load state dict of CNN
    stylegan_finetune_v3.property_score_cnn.load_state_dict(property_cnn_model.state_dict())
    stylegan_finetune_v3.gender_cnn.load_state_dict(gender_cnn_model.state_dict())

    return stylegan_finetune_v3


# 정의된 StyleGAN-FineTune-v3 모델의 Layer 를 Freeze 처리 (CNN은 모두, Generator 는 Freeze 하지 않음)
# Create Date : 2025.04.15
# Last Update Date : 2025.04.16
# - Generator 의 mapping layer 를 제외한 모든 layer 를 Freeze 처리

# Arguments:
# - stylegan_finetune_v3 (nn.Module) : 학습할 StyleGAN-FineTune-v3 모델
# - check_again          (bool)      : freeze 여부 재 확인 테스트용

def freeze_stylegan_finetune_v3_layers(stylegan_finetune_v3, check_again=False):

    # Generator freeze 범위 : Z -> W mapping 을 제외한 모든 레이어
    for name, param in stylegan_finetune_v3.stylegan_generator.named_parameters():
        if name.split('.')[0] != 'mapping':
            param.requires_grad = False

    # 모든 CNN Model freeze 범위 : 전체
    for param in stylegan_finetune_v3.property_score_cnn.parameters():
        param.requires_grad = False

    for param in stylegan_finetune_v3.gender_cnn.parameters():
        param.requires_grad = False

    # 제대로 freeze 되었는지 확인
    if check_again:
        for idx, param in enumerate(stylegan_finetune_v3.stylegan_generator.parameters()):
            print(f'StyleGAN Generator layer {idx} : {param.requires_grad}')

        for idx, param in enumerate(stylegan_finetune_v3.property_score_cnn.parameters()):
            print(f'Property CNN layer {idx} : {param.requires_grad}')

        for idx, param in enumerate(stylegan_finetune_v3.gender_cnn.parameters()):
            print(f'Gender CNN layer {idx} : {param.requires_grad}')


# 정의된 StyleGAN-FineTune-v3 모델을 학습
# Create Date : 2025.04.15
# Last Update Date : 2025.04.17
# - 이미지 생성 테스트를 통과한 z 값이 있는 모델만 저장하도록 수정

# Arguments:
# - stylegan_finetune_v3   (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator         (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator
#                                              (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)
# - fine_tuned_generator_encoder (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator 에 대한 CVAE Encoder

def run_training_stylegan_finetune_v3(stylegan_finetune_v3, fine_tuning_dataloader):
    global mus, log_vars

    stylegan_finetune_v3.train()

    current_epoch = 0
    min_train_loss = None
    min_train_loss_epoch = 0
    best_epoch_model = None

    train_log = {'epoch': [], 'batch_idx': [],
                 'total_loss': [], 'mse': [], 'gender_loss': [], 'mu_loss': [], 'logvar_loss': []}

    for loss_type in loss_types:
        train_log[f'{loss_type}_mse'] = []
    for loss_type in loss_types:
        train_log[f'{loss_type}_abs'] = []

    while True:
        mus = []
        log_vars = []

        train_loss = 0.0
        train_loss_mse = 0.0
        train_loss_gender = 0.0
        train_loss_mu = 0.0
        train_loss_logvar = 0.0

        total = 0
        total_loss_dict = {}
        for loss_type in loss_types:
            total_loss_dict[f'{loss_type}_mse'] = 0.0
            total_loss_dict[f'{loss_type}_abs'] = 0.0

        for idx, raw_data in enumerate(fine_tuning_dataloader):
            is_check = (current_epoch < 10 and idx % 20 == 0) or (current_epoch == 0 and idx < 20)
            use_mu_and_logvar = idx >= len(fine_tuning_dataloader) - IMGS_PER_TEST_PROPERTY_SET
            save_mu_and_logvar = idx >= len(fine_tuning_dataloader) - LAST_BATCHES_TO_SAVE_MU_AND_LOGVAR

            images = raw_data['image']
            images = images.to(stylegan_finetune_v3.device)
            labels = concatenate_property_scores(raw_data)
            labels = labels.to(stylegan_finetune_v3.device).to(torch.float32)

            mu, logvar, gen_img_prop_score, gen_img_gender_score = (
                stylegan_finetune_v3(x=images,
                                     property_label=labels,
                                     tensor_visualize_test=False,
                                     use_mu_and_logvar_for_test_generation=use_mu_and_logvar,
                                     save_mu_and_logvar=save_mu_and_logvar))

            stylegan_finetune_v3.optimizer.zero_grad()

            loss, loss_dict = vae_loss_function(gen_img_prop_score, gen_img_gender_score, labels, mu, logvar)
            loss.backward()

            train_loss_batch = float(loss.detach().cpu().numpy())

            train_loss_batch_mse = loss_dict['mse']
            train_loss_batch_gender = loss_dict['gender_loss']
            train_loss_batch_mu = loss_dict['mu_loss']
            train_loss_batch_logvar = loss_dict['logvar_loss']

            train_loss += train_loss_batch * labels.size(0)
            train_loss_mse += train_loss_batch_mse * labels.size(0)
            train_loss_gender += train_loss_batch_gender * labels.size(0)
            train_loss_mu += train_loss_batch_mu * labels.size(0)
            train_loss_logvar += train_loss_batch_logvar * labels.size(0)

            if is_check:
                print(f'epoch {current_epoch} batch {idx}: loss = {train_loss_batch:.4f}')
                save_train_log(current_epoch, idx, train_log, loss_dict=loss_dict)

            for key in total_loss_dict.keys():
                total_loss_dict[key] += loss_dict[key] * labels.size(0)

            total += labels.size(0)

            stylegan_finetune_v3.optimizer.step()

        train_loss /= total
        train_loss_mse /= total
        train_loss_gender /= total
        train_loss_mu /= total
        train_loss_logvar /= total

        for key in total_loss_dict.keys():
            total_loss_dict[key] /= total
            total_loss_dict[key] = round(total_loss_dict[key], 4)

        total_loss_dict['total_loss'] = train_loss
        total_loss_dict['mse'] = train_loss_mse
        total_loss_dict['gender_loss'] = train_loss_gender
        total_loss_dict['mu_loss'] = train_loss_mu
        total_loss_dict['logvar_loss'] = train_loss_logvar

        # 이미지 생성 테스트
        passed_z_count = test_create_output_images(stylegan_finetune_v3, current_epoch)

        # 로그 출력
        print(f'epoch {current_epoch}: loss = {train_loss:.4f}, passed = {passed_z_count}, '
              f'CUDA memory = {torch.cuda.memory_allocated()}')

        save_train_log(current_epoch, '-', train_log, loss_dict=total_loss_dict)

        # Early Stopping 처리
        if min_train_loss is None or train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_loss_epoch = current_epoch

            best_epoch_model = StyleGANFineTuneV3().to(stylegan_finetune_v3.device)
            best_epoch_model.load_state_dict(stylegan_finetune_v3.state_dict())

        # 모델 저장
        if passed_z_count >= 1:
            passed_model = StyleGANFineTuneV3().to(stylegan_finetune_v3.device)
            passed_model.load_state_dict(stylegan_finetune_v3.state_dict())

            model_dir_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
            ckpt_gen_path = f'{model_dir_path}/stylegan_gen_fine_tuned_v3_ckpt_{current_epoch:04d}_gen.pth'
            ckpt_enc_path = f'{model_dir_path}/stylegan_gen_fine_tuned_v3_ckpt_{current_epoch:04d}_enc.pth'

            torch.save(stylegan_finetune_v3.stylegan_generator.state_dict(), ckpt_gen_path)
            torch.save(stylegan_finetune_v3.CVAE_encoder.state_dict(), ckpt_enc_path)

        if current_epoch >= MAX_EPOCHS and current_epoch - min_train_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1
        stylegan_finetune_v3.scheduler.step()

    fine_tuned_generator = best_epoch_model.stylegan_generator
    fine_tuned_generator_encoder = best_epoch_model.CVAE_encoder

    return fine_tuned_generator, fine_tuned_generator_encoder


# StyleGAN-FineTune-v3 모델 학습 중 Loss 를 csv 로 로깅
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - current_epoch (int)  : 현재 epoch 번호
# - batch_idx     (int)  : 현재 batch 번호
# - train_log     (dict) : 현재 로깅 중인 학습 로그의 dict
# - loss_dict     (dict) : 추가로 로깅할 Loss 의 dict

def save_train_log(current_epoch, batch_idx, train_log, loss_dict):
    train_log_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/train_log_v3.csv'

    train_log['epoch'].append(current_epoch)
    train_log['batch_idx'].append(batch_idx)

    for key in train_log.keys():
        if key in ['epoch', 'batch_idx']:
            continue
        train_log[key].append(loss_dict[key])

    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(train_log_path, index=False)


# StyleGAN-FineTune-v3 모델 학습 중 출력 결과물 테스트
# Create Date : 2025.04.15
# Last Update Date : 2025.04.18
# - 테스트 이미지 생성 로직 수정 및 테스트 합격 기준 수정

# Arguments:
# - stylegan_finetune_v3 (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (StyleGAN-FineTune-v3 으로 Fine-Tuning 중)
# - current_epoch        (int)       : 현재 epoch 의 번호

# Returns:
# - passed_z_count (int) : Oh-LoRA 생성을 위한 z 값으로 합격 판정을 받은 z 값의 개수

def test_create_output_images(stylegan_finetune_v3, current_epoch):
    global mus, log_vars

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_during_finetuning_v3'
    img_save_dir = f'{img_save_dir}/epoch_{current_epoch:04d}'
    os.makedirs(img_save_dir, exist_ok=True)

    # save as DataFrame first
    mus_np = np.round(np.array(mus), 4)
    log_vars_np = np.round(np.array(log_vars), 4)

    mus_torch = torch.tensor(mus[-IMGS_PER_TEST_PROPERTY_SET:])
    log_vars_torch = torch.tensor(log_vars[-IMGS_PER_TEST_PROPERTY_SET:])

    std = torch.exp(0.5 * log_vars_torch)
    eps = torch.randn_like(std)

    z = mus_torch + eps * std
    z = z.to(torch.float32)

    pd.DataFrame(mus_np).to_csv(f'{img_save_dir}/test_mus.csv', index=False)
    pd.DataFrame(log_vars_np).to_csv(f'{img_save_dir}/test_log_vars.csv', index=False)
    pd.DataFrame(np.array(z)).to_csv(f'{img_save_dir}/test_zs.csv', index=False)

    # label: 'eyes', ('hair_color', 'hair_length',) 'mouth', 'pose', ('background_mean', 'background_std')
    eyes_labels = [-1.8, 1.8]
    mouth_labels = [-1.2, -0.6, 0.0, 0.8, 1.6]
    pose_labels = [-1.2, 0.0, 1.2, 2.4, 3.6]

    labels = []
    eyes_label_order = []
    mouth_label_order = []
    pose_label_order = []

    for mouth in mouth_labels:
        for eyes in eyes_labels:
            for pose in pose_labels:
                labels.append([eyes, 0.0, 0.0, mouth, pose, 0.0, 0.0])

                eyes_label_order.append(eyes)
                mouth_label_order.append(mouth)
                pose_label_order.append(pose)

    eyes_corrcoefs, eyes_mses, eyes_maes = [], [], []
    mouth_corrcoefs, mouth_mses, mouth_maes = [], [], []
    pose_corrcoefs, pose_mses, pose_maes = [], [], []

    label_count = len(labels)
    passed_z_count = 0
    passed_check = []

    for z_idx in range(IMGS_PER_TEST_PROPERTY_SET):
        if current_epoch == 0:
            print(f'test generation for z index {z_idx} ...')

        save_dir = f'{img_save_dir}/{z_idx}'
        os.makedirs(save_dir, exist_ok=True)

        eyes_scores = []
        mouth_scores = []
        pose_scores = []

        for label_idx, label in enumerate(labels):
            label_np = np.array([[label]])
            label_np = label_np.reshape((1, PROPERTY_DIMS_Z))
            label_torch = torch.tensor(label_np).to(torch.float32)

            with torch.no_grad():
                z_of_idx = z[z_idx].unsqueeze(0)
                generated_images = stylegan_finetune_v3.stylegan_generator(z=z_of_idx.cuda(),
                                                                           label=label_torch.cuda())['image']
                generated_images = generated_images.detach().cpu()

                property_scores = stylegan_finetune_v3.property_score_cnn(generated_images.cuda())
                property_scores_np = property_scores.detach().cpu().numpy()

                eyes_scores.append(round(property_scores_np[0][0], 4))
                mouth_scores.append(round(property_scores_np[0][3], 4))
                pose_scores.append(round(property_scores_np[0][4], 4))

            save_tensor_png(generated_images[0],
                            image_save_path=f'{save_dir}/test_img_{label_idx:03d}.png')

        eyes_corrcoef = np.corrcoef(eyes_label_order, eyes_scores)[0][1]
        mouth_corrcoef = np.corrcoef(mouth_label_order, mouth_scores)[0][1]
        pose_corrcoef = np.corrcoef(pose_label_order, pose_scores)[0][1]

        eyes_mse = sum((eyes_label_order[i] - eyes_scores[i]) ** 2 for i in range(label_count)) / label_count
        mouth_mse = sum((mouth_label_order[i] - mouth_scores[i]) ** 2 for i in range(label_count)) / label_count
        pose_mse = sum((pose_label_order[i] - pose_scores[i]) ** 2 for i in range(label_count)) / label_count

        eyes_mae = sum(abs(eyes_label_order[i] - eyes_scores[i]) for i in range(label_count)) / label_count
        mouth_mae = sum(abs(mouth_label_order[i] - mouth_scores[i]) for i in range(label_count)) / label_count
        pose_mae = sum(abs(pose_label_order[i] - pose_scores[i]) for i in range(label_count)) / label_count

        eyes_corrcoefs.append(round(eyes_corrcoef, 4))
        mouth_corrcoefs.append(round(mouth_corrcoef, 4))
        pose_corrcoefs.append(round(pose_corrcoef, 4))

        eyes_mses.append(round(eyes_mse, 4))
        mouth_mses.append(round(mouth_mse, 4))
        pose_mses.append(round(pose_mse, 4))

        eyes_maes.append(round(eyes_mae, 4))
        mouth_maes.append(round(mouth_mae, 4))
        pose_maes.append(round(pose_mae, 4))

        # 합격 여부 판정 및 저장
        is_passed_corr = eyes_corrcoef >= 0.77 and mouth_corrcoef >= 0.85 and pose_corrcoef >= 0.82
        is_passed_mae = eyes_mae <= 1.0 and mouth_mae <= 0.6 and pose_mae <= 1.4
        is_passed = is_passed_corr and is_passed_mae

        if is_passed:
            passed_z_count += 1
            passed_check.append('O')
        else:
            passed_check.append('X')

        z_info_dict = {'eyes_score': eyes_scores, 'eyes_label': eyes_label_order,
                       'mouth_score': mouth_scores, 'mouth_label': mouth_label_order,
                       'pose_score': pose_scores, 'pose_label': pose_label_order}
        z_info_df = pd.DataFrame(z_info_dict)
        z_info_df.to_csv(f'{save_dir}/test_result.csv')

    result_dict = {'z_idx': list(range(IMGS_PER_TEST_PROPERTY_SET)), 'passed': passed_check,
                   'eyes_corr': eyes_corrcoefs, 'mouth_corr': mouth_corrcoefs, 'pose_corr': pose_corrcoefs,
                   'eyes_mse': eyes_mses, 'mouth_mse': mouth_mses, 'pose_mse': pose_mses,
                   'eyes_mae': eyes_maes, 'mouth_mae': mouth_maes, 'pose_mae': pose_maes}

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f'{img_save_dir}/result_each_z.csv')

    # 랜덤하게 30장 생성
    z_random_gen = torch.randn((RANDOM_GEN_TEST_IMGS_PER_EPOCH, ORIGINAL_HIDDEN_DIMS_Z)).to(torch.float32)
    label_np = np.random.randn(RANDOM_GEN_TEST_IMGS_PER_EPOCH, PROPERTY_DIMS_Z)
    label_np[:, 6] = 0  # Background std 속성 제외
    label_torch = torch.tensor(label_np).to(torch.float32)

    with torch.no_grad():
        random_generated_images = stylegan_finetune_v3.stylegan_generator(z=z_random_gen.cuda(),
                                                                          label=label_torch.cuda())['image']
        random_generated_images = random_generated_images.detach().cpu()

        gender_scores = stylegan_finetune_v3.gender_cnn(random_generated_images.cuda())
        property_scores = stylegan_finetune_v3.property_score_cnn(random_generated_images.cuda())
        gender_scores = gender_scores.detach().cpu()
        property_scores = property_scores.detach().cpu()

        gender_score_np = gender_scores.numpy()
        property_score_np = property_scores.numpy()

    property_score_info_dict = {
        'img_no': list(range(RANDOM_GEN_TEST_IMGS_PER_EPOCH)),
        'gender_score': list(gender_score_np[:, 0]),
        'eyes_score': list(property_score_np[:, 0]),
        'hair_color_score': list(property_score_np[:, 1]),
        'hair_length_score': list(property_score_np[:, 2]),
        'mouth_score': list(property_score_np[:, 3]),
        'pose_score': list(property_score_np[:, 4]),
        'back_mean_score': list(property_score_np[:, 5]),
        'back_std_score': list(property_score_np[:, 6])
    }
    property_score_info_df = pd.DataFrame(property_score_info_dict)
    property_score_info_df.to_csv(f'{img_save_dir}/test_random_gen_result.csv', index=False)

    for img_no in range(RANDOM_GEN_TEST_IMGS_PER_EPOCH):
        save_tensor_png(random_generated_images[img_no],
                        image_save_path=f'{img_save_dir}/test_random_gen_img_{img_no:03d}.png')

    return passed_z_count


# (테스트용) 매 epoch 마다 생성된 이미지에 대해 학습된 CNN 으로 Property Score 계산
# Create Date : 2025.04.17
# Last Update Date : -

# Arguments:
# - stylegan_finetune_v3 (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (StyleGAN-FineTune-v3 으로 Fine-Tuning)
# - max_epochs           (int)       : 최대 epoch 횟수
# - images_per_epoch     (int)       : epoch 당 이미지 개수

def test_compute_property_score(stylegan_finetune_v3, max_epochs, images_per_epoch=80):
    img_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_during_finetuning_v3'

    for epoch in range(max_epochs):
        print(f'checking epoch {epoch} ...')

        epoch_dir = f'{img_dir}/epoch_{epoch:04d}'

        try:
            property_score_dict = {'epoch': [], 'img_name': [], 'eyes_score': [], 'mouth_score': [], 'pose_score': []}

            image_names = list(filter(lambda x: x.startswith('test_random_gen_img'), os.listdir(epoch_dir)))
            for img_no in range(images_per_epoch):
                image_names.append(f'test_img_{img_no}.png')

            image_paths = [f'{epoch_dir}/{name}' for name in image_names]

            for image_name, image_path in zip(image_names, image_paths):
                image = read_image(image_path)
                image = stylegan_transform(image)

                property_score_dict['epoch'].append(epoch)
                property_score_dict['img_name'].append(image_name)

                with torch.no_grad():
                    property_scores = stylegan_finetune_v3.property_score_cnn(image.unsqueeze(0).cuda())
                    property_score_np = property_scores.detach().cpu().numpy()

                    property_score_dict['eyes_score'].append(round(property_score_np[0][0], 4))
                    property_score_dict['mouth_score'].append(round(property_score_np[0][3], 4))
                    property_score_dict['pose_score'].append(round(property_score_np[0][4], 4))

            property_score_df = pd.DataFrame(property_score_dict)
            property_score_df.to_csv(f'{epoch_dir}/test_property_cnn_result.csv')

        except Exception as e:
            print(f'error: {e}')


# StyleGAN-FineTune-v3 모델 학습
# Create Date : 2025.04.15
# Last Update Date : 2025.04.16
# - Fine-Tuned Generator 의 Encoder 반환 추가

# Arguments:
# - device                 (device)     : 모델을 mapping 시킬 device (GPU 등)
# - generator              (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - property_cnn_model     (nn.Module)  : StyleGAN-FineTune-v3 Fine-Tuning 에 사용할 핵심 속성값 도출용 학습된 CNN 모델
# - gender_cnn_model       (nn.Module)  : StyleGAN-FineTune-v3 Fine-Tuning 에 사용할 성별 판단용 학습된 CNN 모델

# Returns:
# - fine_tuned_generator         (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator
#                                              (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)
# - fine_tuned_generator_encoder (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator 에 대한 CVAE Encoder

def train_stylegan_finetune_v3(device, generator, fine_tuning_dataloader, property_cnn_model, gender_cnn_model):

    # define StyleGAN-FineTune-v3 model
    stylegan_finetune_v3 = define_stylegan_finetune_v3(device, generator, property_cnn_model, gender_cnn_model)
    freeze_stylegan_finetune_v3_layers(stylegan_finetune_v3)

#    test_compute_property_score(stylegan_finetune_v3, max_epochs=70)

    # save model graph of StyleGAN-FineTune-v3 before training
    model_graph = draw_graph(stylegan_finetune_v3,
                             input_data=[torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, 3, IMG_RES, IMG_RES)),
                                         torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, PROPERTY_DIMS_Z))],
                             depth=5)

    visual_graph = model_graph.visual_graph

    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/stylegan_finetune_v3.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    # run Fine-Tuning
    fine_tuned_generator, fine_tuned_generator_encoder = (
        run_training_stylegan_finetune_v3(stylegan_finetune_v3, fine_tuning_dataloader))

    return fine_tuned_generator, fine_tuned_generator_encoder
