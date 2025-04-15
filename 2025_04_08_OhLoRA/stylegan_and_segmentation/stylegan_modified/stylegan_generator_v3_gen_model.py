import torch
from torchview import draw_graph
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

IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 7           # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

CNN_TENSOR_TEST_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/tensor_visualize_test_cnn'
os.makedirs(CNN_TENSOR_TEST_DIR, exist_ok=True)


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
                             input_data=[torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z)),
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

    raise NotImplementedError


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