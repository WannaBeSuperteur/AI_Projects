from generate_dataset.cnn_gender import main_gender
from generate_dataset.cnn_quality import main_quality
from generate_dataset.cnn_age import main_age
from generate_dataset.cnn_glass import main_glass

import os
import pandas as pd
import shutil

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan/generate_dataset'

ORIGINAL_IMAGE_PATH = f'{PROJECT_DIR_PATH}/stylegan/generated_face_images'
COPIED_IMAGE_PATH = f'{PROJECT_DIR_PATH}/stylegan/generated_face_images_filtered'
os.makedirs(COPIED_IMAGE_PATH, exist_ok=True)

TOTAL_IMAGES = 15000


# 모든 이미지 (15,000 장) 에 대한 Gender, Quality 값 (원래 labeling 이 안 된 13,000 장은 그 예측값) 을 모두 취합한 DataFrame 반환
# Create Date : 2025.05.26
# Last Update Date : -

# Files to load:
# - stylegan/generate_dataset/scores_labeled_first_2k.csv      : 첫 2,000 장의 labeling 정보
# - stylegan/generate_dataset/cnn_inference_result/gender.csv  : 나머지 13,000 장의 Gender 속성 값 예측 결과
# - stylegan/generate_dataset/cnn_inference_result/quality.csv : 나머지 13,000 장의 Quality 속성 값 예측 결과
# - stylegan/generate_dataset/cnn_inference_result/age.csv     : 나머지 13,000 장의 Age 속성 값 예측 결과
# - stylegan/generate_dataset/cnn_inference_result/glass.csv   : 나머지 13,000 장의 Glass 속성 값 예측 결과

# Returns:
# - final_collected_data (Pandas DataFrame) : Gender 속성 값에 대한 모델 예측값을 취합하여 저장한 Pandas DataFrame
#                                             columns = ['img_no', 'img_path',
#                                                        'gender_score', 'quality_score', 'age_score', 'glass_score']

def postprocess_all_data():
    labeled_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/scores_labeled_first_2k.csv')
    remaining_gender_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/cnn_inference_result/gender.csv')
    remaining_quality_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/cnn_inference_result/quality.csv')
    remaining_age_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/cnn_inference_result/age.csv')
    remaining_glass_scores_df = pd.read_csv(f'{DATA_DIR_PATH}/cnn_inference_result/glass.csv')

    print('LABELED SCORE :')
    print(labeled_scores_df)

    print('\nREMAINING GENDER SCORE :')
    print(remaining_gender_scores_df)

    print('\nREMAINING QUALITY SCORE :')
    print(remaining_quality_scores_df)

    print('\nREMAINING AGE SCORE :')
    print(remaining_age_scores_df)

    print('\nREMAINING GLASS SCORE :')
    print(remaining_glass_scores_df)

    img_nos = list(range(TOTAL_IMAGES))
    all_paths = [f'{ORIGINAL_IMAGE_PATH}/{i:06d}.jpg' for i in range(TOTAL_IMAGES)]

    labeled_quality = labeled_scores_df['quality'].tolist()
    labeled_gender = labeled_scores_df['gender'].tolist()
    labeled_age = labeled_scores_df['age'].tolist()
    labeled_glass = labeled_scores_df['glass'].tolist()

    remaining_quality = remaining_quality_scores_df['property_quality_final_score'].tolist()
    remaining_gender = remaining_gender_scores_df['property_gender_final_score'].tolist()
    remaining_age = remaining_gender_scores_df['property_age_final_score'].tolist()
    remaining_glass = remaining_gender_scores_df['property_glass_final_score'].tolist()

    all_quality = labeled_quality + remaining_quality
    all_gender = labeled_gender + remaining_gender
    all_age = labeled_age + remaining_age
    all_glass = labeled_glass + remaining_glass

    final_collected_data_dict = {'img_no': img_nos,
                                 'img_path': all_paths,
                                 'gender_score': all_gender,
                                 'quality_score': all_quality,
                                 'age_score': all_age,
                                 'glass_score': all_glass}

    final_collected_data = pd.DataFrame(final_collected_data_dict)

    return final_collected_data


# 최종 필터링된 이미지를 학습 데이터셋 디렉토리에 복사 (score threshold 는 epoch 별 detail 의 성능지표 계산 시와 다를 수 있음)
# Create Date : 2025.05.26
# Last Update Date : -

# Arguments:
# - final_df (Pandas DataFrame) : Gender, Quality 값을 모두 취합한 Pandas DataFrame

# Returns:
# - 없음

def copy_to_training_data(final_df):
    gender_thrsh = 0.7
    quality_thrsh = 0.9
    age_thrsh = 0.7
    glass_thrsh = 0.05

    filtered_df = final_df[(final_df['gender_score'] >= gender_thrsh) & (final_df['quality_score'] >= quality_thrsh) &
                           (final_df['age_score'] <= age_thrsh) & (final_df['glass_sore'] <= glass_thrsh)]

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

    # Age 예측값 생성 (CNN 학습 포함)
    main_age()

    # Glass 예측값 생성 (CNN 학습 포함)
    main_glass()

    # 최종 취합
    final_collected_data = postprocess_all_data()
    final_collected_data_path = f'{PROJECT_DIR_PATH}/stylegan/generate_dataset/scores_all_15k.csv'
    final_collected_data.to_csv(final_collected_data_path)

    # 필터링된 이미지만 새로운 디렉토리에 복사
    copy_to_training_data(final_collected_data)
