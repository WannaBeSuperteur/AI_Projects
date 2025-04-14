
import os
import torch
import numpy as np
import pandas as pd


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
PROPERTY_SCORE_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/property_score_results'
PROPERTY_SCORE_COMPARE_DIR = f'{PROPERTY_SCORE_DIR}/compare'

os.makedirs(PROPERTY_SCORE_COMPARE_DIR, exist_ok=True)


# CNN 모델에 의한 Property 값 생성
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - cnn_model              (nn.Module)  : 학습된 CNN 모델

# Returns:
# - cnn_property (Pandas DataFrame) : CNN 모델에 의해 생성된 Property 값을 저장한 DataFrame

def compute_cnn_property_values(fine_tuning_dataloader, cnn_model):

    cnn_property_dict = {'img_no': [], 'img_path': [],
                         'eyes_score': [], 'hair_color_score': [], 'hair_length_score': [],
                         'mouth_score': [], 'pose_score': [],
                         'background_mean_score': [], 'background_std_score': []}

    with torch.no_grad():
        for idx, raw_data in enumerate(fine_tuning_dataloader):
            if idx % 25 == 0:
                print(idx)

            images = raw_data['image']
            image_paths = list(raw_data['img_path'])
            image_nos = [int(path.split('/')[1].split('.')[0]) for path in image_paths]
            image_full_paths = [f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/{path}' for path in image_paths]

            images = images.to(cnn_model.device)
            outputs = cnn_model(images).to(torch.float32)
            outputs_np = outputs.detach().cpu().numpy()

            cnn_property_dict['img_no'] += image_nos
            cnn_property_dict['img_path'] += image_full_paths

            cnn_property_dict['eyes_score'] += list(np.round(outputs_np[:, 0], 4))
            cnn_property_dict['hair_color_score'] += list(np.round(outputs_np[:, 1], 4))
            cnn_property_dict['hair_length_score'] += list(np.round(outputs_np[:, 2], 4))
            cnn_property_dict['mouth_score'] += list(np.round(outputs_np[:, 3], 4))
            cnn_property_dict['pose_score'] += list(np.round(outputs_np[:, 4], 4))
            cnn_property_dict['background_mean_score'] += list(np.round(outputs_np[:, 5], 4))
            cnn_property_dict['background_std_score'] += list(np.round(outputs_np[:, 6], 4))

    cnn_property = pd.DataFrame(cnn_property_dict)
    cnn_property = cnn_property.sort_values(by=['img_no'], axis=0)
    return cnn_property


# CNN 모델에 의한 Property 값과 기존 for StyleGAN-FineTune-v2,v3 Property 값의 비교 결과 생성
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - cnn_property (Pandas DataFrame) : CNN 모델에 의해 생성된 Property 값을 저장한 DataFrame

# Returns:
# - segmentation/property_score_results/compare/all_scores_v2_vs_cnn.csv 에 비교 결과 생성

def compute_property_values(cnn_property):
    print(cnn_property)

    raise NotImplementedError


# for StyleGAN-FineTune-v2,v3 Property 값 (segmentation/property_score_results/all_scores_v2.csv) 와
# CNN 모델이 생성한 Property 값 비교 결과 생성

# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - fine_tuning_dataloader (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader
# - cnn_model              (nn.Module)  : 학습된 CNN 모델

# Returns:
# - segmentation/property_score_results/all_scores_v2_cnn.csv 에 CNN 모델이 생성한 Property 값 저장
# - segmentation/property_score_results/compare/all_scores_v2_vs_cnn.csv 에 비교 결과 생성

def write_property_compare_result(fine_tuning_dataloader, cnn_model):
    cnn_property = compute_cnn_property_values(fine_tuning_dataloader, cnn_model)
    cnn_property.to_csv(f'{PROPERTY_SCORE_DIR}/all_scores_v2_cnn.csv', index=False)

    compute_property_values(cnn_property)
