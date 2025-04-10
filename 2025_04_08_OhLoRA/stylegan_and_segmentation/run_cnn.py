from cnn.cnn_gender import main_gender
from cnn.cnn_quality import main_quality

import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# 모든 이미지 (10,000 장) 에 대한 Gender, Quality 값 (원래 labeling 이 안 된 8,000 장은 그 예측값) 을 모두 취합한 DataFrame 반환
# Create Date : 2025.04.10
# Last Update Date : -

# Files to load:
# - stylegan_and_segmentation/cnn/synthesize_results_quality_and_gender.csv : 첫 2,000 장의 labeling 정보
# - stylegan_and_segmentation/cnn/inference_result/gender.csv               : 나머지 8,000 장의 Gender 속성 값 예측 결과
# - stylegan_and_segmentation/cnn/inference_result/quality.csv              : 나머지 8,000 장의 Quality 속성 값 예측 결과

# Returns:
# - final_collected_data (Pandas DataFrame) : Gender 속성 값에 대한 모델 예측값을 취합하여 저장한 Pandas DataFrame
#                                             columns = ['img_no', 'img_path', 'gender_score', 'quality_score']

def postprocess_all_data():
    raise NotImplementedError


if __name__ == '__main__':

    # Quality 예측값 생성 (CNN 학습 포함)
    main_quality()

    # Gender 예측값 생성 (CNN 학습 포함)
    main_gender()

    # 최종 취합
    final_collected_data = postprocess_all_data()
    final_collected_data_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/cnn/all_image_quality_and_gender.csv'
    final_collected_data.to_csv(final_collected_data_path)
