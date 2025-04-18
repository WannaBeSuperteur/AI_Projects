import torch
import os
import sys
import numpy as np
import pandas as pd

from stylegan_modified.stylegan_generator import StyleGANGeneratorForV3
from stylegan_modified.stylegan_generator_v2 import load_cnn_model
from stylegan_modified.stylegan_generator_v3_gen_model import train_stylegan_finetune_v3
from cnn.cnn_gender import GenderCNN

global_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
sys.path.append(global_path)

from global_common.visualize_tensor import save_tensor_png


torch.set_printoptions(linewidth=160, sci_mode=False)
np.set_printoptions(suppress=True)


IMG_RESOLUTION = 256
PROPERTY_DIMS_Z = 7  # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator 불러오기
# Create Date : 2025.04.15
# Last Update Date : 2025.04.18
# - StyleGAN-FineTune-v3 Generator 와 함께 z vector 정보가 있을 시, 해당 z vector 로 테스트 이미지 생성 & 해당 generator 를 반환

# Arguments:
# - v3_gen_path (str)    : StyleGAN-FineTune-v3 모델 저장 경로
# - device      (device) : StyleGAN-FineTune-v3 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator

def load_stylegan_finetune_v3_model(v3_gen_path, device):
    try:
        fine_tuned_generator = StyleGANGeneratorForV3(resolution=IMG_RESOLUTION)
        fine_tuned_generator.load_state_dict(torch.load(v3_gen_path, map_location=device, weights_only=False))

        fine_tuned_generator.to(device)
        fine_tuned_generator.device = device

    except Exception as e:
        print(f'FINAL StyleGAN-FineTune-v3 load failed : {e} / try loading checkpoint ...')

        stylegan_modified_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
        ckpt_gen_names = list(filter(lambda x: x.startswith('stylegan_gen_fine_tuned_v3_ckpt') and x.endswith('gen.pth'),
                                     os.listdir(stylegan_modified_dir)))
        ckpt_gen_paths = [f'{stylegan_modified_dir}/{name}' for name in ckpt_gen_names]
        ckpt_gen_epoch_nos = [int(name.split('_')[6]) for name in ckpt_gen_names]

        assert len(ckpt_gen_paths) >= 1, "There is no checkpoint StyleGAN-FineTune-v3 generator model."

        for ckpt_gen_path, epoch_no in zip(ckpt_gen_paths, ckpt_gen_epoch_nos):
            print(f'testing checkpoint at epoch {epoch_no} ...')

            fine_tuned_generator_ckpt = StyleGANGeneratorForV3(resolution=IMG_RESOLUTION)
            fine_tuned_generator_ckpt.to(device)
            fine_tuned_generator_ckpt.device = device

            fine_tuned_generator_ckpt.load_state_dict(torch.load(ckpt_gen_path,
                                                                 map_location=device,
                                                                 weights_only=False))

            run_ckpt_gen_test(stylegan_generator=fine_tuned_generator_ckpt,
                              epoch_no=epoch_no)

        # return last (largest epoch value) checkpoint generator
        print(f'Test finished successfully!! Return is generator at epoch {ckpt_gen_epoch_nos[-1]}. 😊')
        return fine_tuned_generator_ckpt

    return fine_tuned_generator


# StyleGAN-FineTune-v3 의 Checkpoint Model 에 대한 이미지 생성 테스트
# Create Date : 2025.04.18
# Last Update Date : -

# Arguments:
# - stylegan_generator (nn.Module) : StyleGAN-FineTune-v3 Checkpoint 모델의 Image Generator (= CVAE Decoder)
# - epoch_no           (int)       : 해당 Checkpoint 모델이 생성된 epoch 번호 (0부터 시작)

def run_ckpt_gen_test(stylegan_generator, epoch_no):

    # 생성할 이미지의 Property Label 지정
    eyes_labels = [-1.8, -0.9, 0.0, 0.9, 1.8]
    mouth_labels = [-1.2, -0.6, 0.0, 0.8, 1.6]
    pose_labels = [-1.2, 0.0, 1.2, 2.4, 3.6]

    # z vector 로딩
    stylegan_modified_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    z_vector_csv_path = f'{stylegan_modified_dir}/test_z_vector_{epoch_no:04d}.csv'

    z_vector = pd.read_csv(z_vector_csv_path)
    z_vector_torch = torch.tensor(np.array(z_vector))

    for eyes_label_idx, eyes_label in enumerate(eyes_labels):
        img_save_path = f'{stylegan_modified_dir}/inference_test_v3/ckpt_at_{epoch_no}/eyes_label_{eyes_label_idx}'
        os.makedirs(img_save_path, exist_ok=True)

        for mouth_label_idx, mouth_label in enumerate(mouth_labels):
            for pose_label_idx, pose_label in enumerate(pose_labels):
                label = [eyes_label, 0.0, 0.0, mouth_label, pose_label, 0.0, 0.0]

                label_np = np.array([[label]])
                label_np = label_np.reshape((1, PROPERTY_DIMS_Z))
                label_torch = torch.tensor(label_np).to(torch.float32)

                with torch.no_grad():
                    z = z_vector_torch.to(torch.float32)
                    z_noised = z + 0.3 * torch.randn_like(z)

                    generated_image = stylegan_generator(z=z.cuda(), label=label_torch.cuda())['image']
                    generated_image = generated_image.detach().cpu()

                    generated_image_noised = stylegan_generator(z=z_noised.cuda(), label=label_torch.cuda())['image']
                    generated_image_noised = generated_image_noised.detach().cpu()

                save_tensor_png(generated_image[0],
                                image_save_path=f'{img_save_path}/test_m{mouth_label_idx}_p{pose_label_idx}.png')

                save_tensor_png(generated_image_noised[0],
                                image_save_path=f'{img_save_path}/test_m{mouth_label_idx}_p{pose_label_idx}_noised.png')


# 학습된 성별 판단용 CNN 모델 불러오기
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - gender_cnn_model_path (str)    : 성별 판단용 CNN 모델 저장 경로
# - device                (device) : 성별 판단용 CNN 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - gender_cnn_model (nn.Module) : 학습된 성별 판단용 CNN 모델

def load_gender_cnn_model(gender_cnn_model_path, device):
    cnn_model = GenderCNN()
    cnn_model.load_state_dict(torch.load(gender_cnn_model_path, map_location=device, weights_only=False))

    cnn_model.to(device)
    cnn_model.device = device

    return cnn_model


# StyleGAN-FineTune-v3 모델 Fine-Tuning 실시
# Create Date : 2025.04.15
# Last Update Date : 2025.04.16
# - Fine-Tuned Generator 의 Encoder 추가 저장

# Arguments:
# - generator              (nn.Module)  : StyleGAN-FineTune-v1 모델의 Generator
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v3 모델의 Generator
# - exist_dict           (dict)      : 각 모델 (CNN, StyleGAN-FineTune-v3) 의 존재 여부 (= 신규 학습 미 실시 여부)

def run_fine_tuning(generator, fine_tuning_dataloader):
    stylegan_finetune_v3_exist = False
    stylegan_modified_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v3 : {device}')

    # load pre-trained CNN model for property and gender
    property_cnn_save_path = f'{stylegan_modified_path}/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_cnn_model = load_cnn_model(property_cnn_save_path, device)

    gender_cnn_save_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/models/gender_model_0.pt'
    gender_cnn_model = load_gender_cnn_model(gender_cnn_save_path, device)

    # load or newly train Fine-Tuned Generator (StyleGAN-FineTune-v3)
    v3_gen_path = f'{stylegan_modified_path}/stylegan_gen_fine_tuned_v3.pth'
    v3_gen_encoder_path = f'{stylegan_modified_path}/stylegan_gen_fine_tuned_v3_encoder.pth'

    try:
        fine_tuned_generator = load_stylegan_finetune_v3_model(v3_gen_path, device)
        stylegan_finetune_v3_exist = True

    except Exception as e:
        print(f'StyleGAN-FineTune-v3 model load (or test checkpoint generator model) failed : {e}')

        # train StyleGAN-FineTune-v3 model
        fine_tuned_generator, fine_tuned_generator_encoder = train_stylegan_finetune_v3(device,
                                                                                        generator,
                                                                                        fine_tuning_dataloader,
                                                                                        property_cnn_model,
                                                                                        gender_cnn_model)

        torch.save(fine_tuned_generator.state_dict(), v3_gen_path)
        torch.save(fine_tuned_generator_encoder.state_dict(), v3_gen_encoder_path)

    exist_dict = {'stylegan_finetune_v3': stylegan_finetune_v3_exist}
    print(f'model existance : {exist_dict}')

    return fine_tuned_generator, exist_dict
