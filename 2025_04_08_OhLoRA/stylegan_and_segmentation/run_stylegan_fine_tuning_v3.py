import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

from run_stylegan_fine_tuning import PROPERTY_SCORE_DIR_PATH, ORIGINAL_HIDDEN_DIMS_Z, PROPERTY_DIMS_Z
from run_stylegan_fine_tuning import (PropertyScoreImageDataset,
                                      stylegan_transform,
                                      save_model_structure_pdf,
                                      freeze_generator_layers,
                                      print_summary)

from run_stylegan_fine_tuning_v2 import load_existing_stylegan_state_dict, create_stylegan_finetune_v1

import stylegan_modified.stylegan_generator_inference as modified_inf
from stylegan_modified.stylegan_generator_v3 import run_fine_tuning


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_BATCH_SIZE = 8


# StyleGAN Fine-Tuning 용 데이터셋 (CNN 에 의해 계산된 property scores) 의 Data Loader 로딩
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - stylegan_ft_loader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

def get_stylegan_fine_tuning_dataloader():
    property_score_csv_path = f'{PROPERTY_SCORE_DIR_PATH}/all_scores_v2_cnn.csv'
    property_score_df = pd.read_csv(property_score_csv_path)

    stylegan_ft_dataset = PropertyScoreImageDataset(dataset_df=property_score_df, transform=stylegan_transform)
    stylegan_ft_loader = DataLoader(stylegan_ft_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    return stylegan_ft_loader


# StyleGAN-Fine-Tune-v1 -> v3 으로의 Fine-Tuning 이전 inference test 실시
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator

# Returns:
# - stylegan_modified/inference_test_before_finetuning_v3 에 생성 결과 저장

def run_inference_test_before_finetuning(generator):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_before_finetuning_v3'
    modified_inf.synthesize(generator, num=20, save_dir=img_save_dir, z=None, label=None)


# StyleGAN-FineTune-v3 모델의 Fine-Tuning 실시 (핵심 속성 값 7개를 latent vector 에 추가)
# Create Date : 2025.04.15
# Last Update Date : -

# Arguments:
# - generator_state_dict   (OrderedDict) : StyleGAN-FineTune-v1 모델의 Generator 의 state_dict
# - fine_tuning_dataloader (DataLoader)  : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - device                 (device)      : StyleGAN-FineTune-v3 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - stylegan_modified/stylegan_gen_fine_tuned_v3.pth 에 Fine-Tuning 된 StyleGAN-FineTune-v3 의 Generator 모델 저장

def run_stylegan_fine_tuning(generator_state_dict, fine_tuning_dataloader, device):

    # StyleGAN-FineTune-v1 모델 생성
    generator = create_stylegan_finetune_v1(generator_state_dict, device)

    # 모델 구조를 PDF 로 저장
    save_model_structure_pdf(generator,
                             model_name='stylegan_finetune_v1_generator',
                             input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)])

    # StyleGAN-FineTune-v1 모델의 레이어 freeze 처리
    freeze_generator_layers(generator)
    print_summary(generator,
                  model_name='stylegan_finetune_v1_generator',
                  input_size=[(TRAIN_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                              (TRAIN_BATCH_SIZE, PROPERTY_DIMS_Z)],
                  print_frozen=True)

    # fine tuning 이전 inference 테스트
    run_inference_test_before_finetuning(generator)

    # fine tuning 실시
    fine_tuned_generator, fine_tuned_generator_cnn, exist_dict = run_fine_tuning(generator, fine_tuning_dataloader)

    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    os.makedirs(fine_tuned_model_path, exist_ok=True)

    if not exist_dict['stylegan_finetune_v3']:
        torch.save(fine_tuned_generator.state_dict(), f'{fine_tuned_model_path}/stylegan_gen_fine_tuned_v3.pth')
        print('StyleGAN-FineTune-v3 saved')


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN-FineTune-v3 : {device}')

    # load Pre-trained StyleGAN
    generator_state_dict = load_existing_stylegan_state_dict(device)

    # load DataLoader
    fine_tuning_dataloader = get_stylegan_fine_tuning_dataloader()

    # Fine Tuning
    run_stylegan_fine_tuning(generator_state_dict, fine_tuning_dataloader, device)
