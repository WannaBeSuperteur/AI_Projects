from cnn.cnn_gender import main_gender
from cnn.cnn_quality import main_quality

import os
import pandas as pd
import shutil

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn'

ORIGINAL_IMAGE_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results'
COPIED_IMAGE_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
os.makedirs(COPIED_IMAGE_PATH, exist_ok=True)

TOTAL_IMAGES = 10000


# 모든 이미지 (10,000 장) 에 대한 Gender, Quality 값 (원래 labeling 이 안 된 8,000 장은 그 예측값) 을 모두 취합한 DataFrame 반환
# Create Date : 2025.04.11
# Last Update Date : -

# Files to load:
# - stylegan_and_segmentation/cnn/synthesize_results_quality_and_gender.csv : 첫 2,000 장의 labeling 정보
# - stylegan_and_segmentation/cnn/inference_result/gender.csv               : 나머지 8,000 장의 Gender 속성 값 예측 결과
# - stylegan_and_segmentation/cnn/inference_result/quality.csv              : 나머지 8,000 장의 Quality 속성 값 예측 결과

# Returns:
# - final_collected_data (Pandas DataFrame) : Gender 속성 값에 대한 모델 예측값을 취합하여 저장한 Pandas DataFrame
#                                             columns = ['img_no', 'img_path', 'gender_score', 'quality_score']

def postprocess_all_data():
    labeled_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/synthesize_results_quality_and_gender.csv')
    remaining_gender_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/inference_result/gender.csv')
    remaining_quality_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/inference_result/quality.csv')

    print('LABELED SCORE :')
    print(labeled_scores_df)

    print('\nREMAINING GENDER SCORE :')
    print(remaining_gender_scores_df)

    print('\nREMAINING QUALITY SCORE :')
    print(remaining_quality_scores_df)

    img_nos = list(range(TOTAL_IMAGES))
    all_paths = [f'{ORIGINAL_IMAGE_PATH}/{i:06d}.jpg' for i in range(TOTAL_IMAGES)]

    labeled_quality = labeled_scores_df['quality'].tolist()
    labeled_gender = labeled_scores_df['gender'].tolist()
    remaining_quality = remaining_quality_scores_df['property_quality_final_score'].tolist()
    remaining_gender = remaining_gender_scores_df['property_gender_final_score'].tolist()

    all_quality = labeled_quality + remaining_quality
    all_gender = labeled_gender + remaining_gender

    final_collected_data_dict = {'img_no': img_nos,
                                 'img_path': all_paths,
                                 'gender_score': all_gender,
                                 'quality_score': all_quality}

    final_collected_data = pd.DataFrame(final_collected_data_dict)

    return final_collected_data


# 최종 필터링된 이미지를 학습 데이터셋 디렉토리에 복사 (score threshold 는 epoch 별 detail 의 성능지표 계산 시와 다를 수 있음)
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - final_df (Pandas DataFrame) : Gender, Quality 값을 모두 취합한 Pandas DataFrame

# Returns:
# - 없음

def copy_to_training_data(final_df):
    gender_thrsh = 0.7
    quality_thrsh = 0.9

    filtered_df = final_df[(final_df['gender_score'] >= gender_thrsh) & (final_df['quality_score'] >= quality_thrsh)]
    print('FILTERED DATA :')
    print(filtered_df)

    filtered_paths = filtered_df['img_path'].tolist()
    filtered_nos = filtered_df['img_no'].tolist()

    for original_img_no, original_img_path in zip(filtered_nos, filtered_paths):
        dest_path = f'{COPIED_IMAGE_PATH}/{original_img_no:06d}.jpg'
        shutil.copy(original_img_path, dest_path)


if __name__ == '__main__':

    # Quality 예측값 생성 (CNN 학습 포함)
    main_quality()

    # Gender 예측값 생성 (CNN 학습 포함)
    main_gender()

    # 최종 취합
    final_collected_data = postprocess_all_data()
    final_collected_data_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/all_image_quality_and_gender.csv'
    final_collected_data.to_csv(final_collected_data_path)

    # 필터링된 이미지만 새로운 디렉토리에 복사
    copy_to_training_data(final_collected_data)
