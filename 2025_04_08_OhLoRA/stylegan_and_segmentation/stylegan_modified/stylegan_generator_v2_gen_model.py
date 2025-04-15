import torch
import torch.nn as nn
from torchview import draw_graph
import pandas as pd
import numpy as np

from stylegan_modified.stylegan_generator import StyleGANGeneratorForV2
from stylegan_modified.stylegan_generator_v2_cnn import PropertyScoreCNN

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
EARLY_STOPPING_ROUNDS = 999999  # for test
STEP_GROUP_SIZE = 10            # for test

IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 7           # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

CNN_TENSOR_TEST_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/tensor_visualize_test_cnn'
os.makedirs(CNN_TENSOR_TEST_DIR, exist_ok=True)


class StyleGANFineTuneV2(nn.Module):
    def __init__(self):
        super(StyleGANFineTuneV2, self).__init__()

        self.stylegan_generator = StyleGANGeneratorForV2(resolution=IMAGE_RESOLUTION)
        self.property_score_cnn = PropertyScoreCNN()

        kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
        self.stylegan_generator.G_kwargs_val = kwargs_val

    def forward(self, z, property_label, tensor_visualize_test=False):
        generated_image = self.stylegan_generator(z, property_label, style_mixing_prob=0.0)
        property_score = self.property_score_cnn(generated_image['image'])

        if tensor_visualize_test:
            test_name = 'test_before_finetune'

            current_batch_size = generated_image['image'].size(0)
            property_score_np = property_score.detach().cpu().numpy()

            property_score_info_dict = {
                'img_no': list(range(current_batch_size)),
                'eyes_score': list(property_score_np[:, 0]),
                'hair_color_score': list(property_score_np[:, 1]),
                'hair_length_score': list(property_score_np[:, 2]),
                'mouth_score': list(property_score_np[:, 3]),
                'pose_score': list(property_score_np[:, 4]),
                'back_mean_score': list(property_score_np[:, 5]),
                'back_std_score': list(property_score_np[:, 6])
            }
            property_score_info_df = pd.DataFrame(property_score_info_dict)
            property_score_info_df.to_csv(f'{CNN_TENSOR_TEST_DIR}/finetune_v2_{test_name}_result.csv',
                                          index=False)

            for i in range(current_batch_size):
                save_tensor_png(generated_image['image'][i],
                                image_save_path=f'{CNN_TENSOR_TEST_DIR}/finetune_v2_{test_name}_{i:03d}.png')

        return property_score


# StyleGAN-FineTune-v2 모델 정의 및 generator 의 state_dict 를 로딩
# Create Date : 2025.04.14
# Last Update Date : 2025.04.15
# - AdamW optimizer 적용 대상 parameter 지정 오류 수정
# - Learning rate 5e-5 -> 1e-5 로 수정

# Arguments:
# - device    (device)    : 모델을 mapping 시킬 device (GPU 등)
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - cnn_model (nn.Module) : StyleGAN-FineTune-v2 Fine-Tuning 에 사용할 학습된 CNN 모델

# Returns:
# - stylegan_finetune_v2 (nn.Module) : 학습할 StyleGAN-FineTune-v2 모델

def define_stylegan_finetune_v2(device, generator, cnn_model):
    stylegan_finetune_v2 = StyleGANFineTuneV2()
    stylegan_finetune_v2.optimizer = torch.optim.AdamW(stylegan_finetune_v2.parameters(), lr=0.00001)
    stylegan_finetune_v2.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=stylegan_finetune_v2.optimizer,
                                                                                T_max=20,
                                                                                eta_min=0)

    stylegan_finetune_v2.to(device)
    stylegan_finetune_v2.device = device

    # load state dict of generator and CNN model
    stylegan_finetune_v2.stylegan_generator.load_state_dict(generator.state_dict())
    stylegan_finetune_v2.property_score_cnn.load_state_dict(cnn_model.state_dict())

    # save model graph of StyleGAN-FineTune-v2 before training
    model_graph = draw_graph(stylegan_finetune_v2,
                             input_data=[torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z)),
                                         torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, PROPERTY_DIMS_Z))],
                             depth=5)

    visual_graph = model_graph.visual_graph

    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/stylegan_finetune_v2.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    return stylegan_finetune_v2


# 정의된 StyleGAN-FineTune-v2 모델의 Layer 를 Freeze 처리 (CNN은 모두, Generator 는 Dense Layer 제외 모두)
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - stylegan_finetune_v2 (nn.Module) : 학습할 StyleGAN-FineTune-v2 모델
# - cnn_model            (nn.Module) : StyleGAN-FineTune-v2 Fine-Tuning 에 사용할 학습된 CNN 모델
# - check_again          (bool)      : freeze 여부 재 확인 테스트용

def freeze_stylegan_finetune_v2_layers(stylegan_finetune_v2, cnn_model, check_again=False):

    # StyleGAN-FineTune-v2 freeze 범위 : Z -> W mapping 을 제외한 모든 레이어
    for name, param in stylegan_finetune_v2.named_parameters():
        if name.split('.')[1] != 'mapping':
            param.requires_grad = False

    # CNN Model freeze 범위 : 전체
    for name, param in cnn_model.named_parameters():
        param.requires_grad = False

    # 제대로 freeze 되었는지 확인
    if check_again:
        for idx, param in enumerate(stylegan_finetune_v2.parameters()):
            print(f'StyleGAN-FineTune-v2 layer {idx} : {param.requires_grad}')

        for idx, param in enumerate(cnn_model.parameters()):
            print(f'CNN Model layer {idx} : {param.requires_grad}')


# 정의된 StyleGAN-FineTune-v2 모델을 학습
# Create Date : 2025.04.14
# Last Update Date : 2025.04.15
# - 각 속성 별 Loss 출력 추가

# Arguments:
# - stylegan_finetune_v2 (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator
#                                      (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)

def run_training_stylegan_finetune_v2(stylegan_finetune_v2):
    stylegan_finetune_v2.train()

    current_step_group = 0
    smallest_loss = None
    smallest_loss_step_group = 0
    stylegan_finetune_v2_at_best_step_group = None

    print('Fine-Tuning StyleGAN-FineTune-v2 start.')

    while True:
        step_group_loss = 0.0
        property_loss_dict = {'eyes': 0.0, 'hair_color': 0.0, 'hair_length': 0.0,
                              'mouth': 0.0, 'pose': 0.0, 'background_mean': 0.0}

        for _ in range(STEP_GROUP_SIZE):
            z = torch.randn((TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z))
            z = z.to(stylegan_finetune_v2.device)

            property_label = torch.randn((TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z))
            property_label = property_label.to(stylegan_finetune_v2.device)

            # train 실시
            stylegan_finetune_v2.optimizer.zero_grad()
            property_output = stylegan_finetune_v2(z=z, property_label=property_label).to(torch.float32)

            loss = nn.MSELoss()(property_output[:, :6], property_label[:, :6])  # Background Std score 를 제외한 Loss
            loss.backward()
            stylegan_finetune_v2.optimizer.step()

            loss_float = float(loss.detach().cpu().numpy())
            step_group_loss += loss_float

            # 개별 property 의 Loss 계산
            for idx, property in enumerate(property_loss_dict.keys()):
                property_loss = nn.MSELoss()(property_output[:, idx:idx+1], property_label[:, idx:idx+1])
                property_loss_dict[property] += float(property_loss.detach().cpu().numpy())

        # Loss 출력
        step_group_loss /= STEP_GROUP_SIZE
        for key in property_loss_dict.keys():
            property_loss_dict[key] /= STEP_GROUP_SIZE
            property_loss_dict[key] = round(property_loss_dict[key], 4)

        print(f'step group {current_step_group} (best: {smallest_loss_step_group}), '
              f'current loss: {step_group_loss:.4f} ({property_loss_dict})')

        # Early Stopping 처리
        if smallest_loss is None or step_group_loss < smallest_loss:
            smallest_loss = step_group_loss
            smallest_loss_step_group = current_step_group

            stylegan_finetune_v2_at_best_step_group = StyleGANFineTuneV2().to(stylegan_finetune_v2.device)
            stylegan_finetune_v2_at_best_step_group.load_state_dict(stylegan_finetune_v2.state_dict())

        if current_step_group - smallest_loss_step_group >= EARLY_STOPPING_ROUNDS:
            break

        # 정보 갱신 및 이미지 생성 테스트
        stylegan_finetune_v2.scheduler.step()
        test_create_output_images(stylegan_finetune_v2, current_step_group)
        current_step_group += 1

    fine_tuned_generator = stylegan_finetune_v2_at_best_step_group.stylegan_generator
    return fine_tuned_generator


# StyleGAN-FineTune-v2 모델 학습 중 출력 결과물 테스트
# Create Date : 2025.04.14
# Last Update Date : 2025.04.15
# - Memory Leak 해결

# Arguments:
# - stylegan_finetune_v2 (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - current_step_group   (int)       : 현재 step group 의 번호

def test_create_output_images(stylegan_finetune_v2, current_step_group):
    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_during_finetuning_v2'
    img_save_dir = f'{img_save_dir}/step_group_{current_step_group:04d}'
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
            generated_images = stylegan_finetune_v2.stylegan_generator(z=z.cuda(), label=label_torch.cuda())['image']
            generated_images = generated_images.detach().cpu()
        image_count = generated_images.size(0)

        for img_idx in range(image_count):
            img_no = label_idx * IMGS_PER_TEST_PROPERTY_SET + img_idx

            save_tensor_png(generated_images[img_idx],
                            image_save_path=f'{img_save_dir}/test_img_{img_no}.png')


# StyleGAN-FineTune-v2 모델 학습
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - device    (device)    : 모델을 mapping 시킬 device (GPU 등)
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - cnn_model (nn.Module) : StyleGAN-FineTune-v2 Fine-Tuning 에 사용할 학습된 CNN 모델

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator
#                                      (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)

def train_stylegan_finetune_v2(device, generator, cnn_model):

    # define StyleGAN-FineTune-v2 model
    stylegan_finetune_v2 = define_stylegan_finetune_v2(device, generator, cnn_model)
#    freeze_stylegan_finetune_v2_layers(stylegan_finetune_v2, cnn_model)

    # run Fine-Tuning
    fine_tuned_generator = run_training_stylegan_finetune_v2(stylegan_finetune_v2)

    return fine_tuned_generator
