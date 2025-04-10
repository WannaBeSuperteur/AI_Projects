
from cnn.cnn_common import (load_dataset,
                            load_remaining_images_dataset,
                            load_cnn_model,
                            train_cnn_models,
                            predict_score_image)

import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
inference_result_dir = f'{PROJECT_DIR_PATH}/cnn/inference_result'


# labeling 이 안 된 8,000 장에 대해 예측된 Gender 속성 값 반환
# Create Date : 2025.04.10
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - final_score (Pandas DataFrame) : Gender 속성 값에 대한 모델 예측값을 저장한 Pandas DataFrame
#                                    columns = ['img_no', 'img_path', 'property_gender_final_score',
#                                               'score_model_0', 'score_model_1', ...]

def main_gender():

    # load dataset
    data_loader = load_dataset(property_name='gender')

    # load or train model
    try:
        print('loading CNN models ...')
        cnn_models = load_cnn_model(property_name='gender')
        print('loading CNN models successful!')

    except Exception as e:
        print(f'CNN model load failed : {e}')
        cnn_models = train_cnn_models(data_loader)

    # performance evaluation
    remaining_image_loader = load_remaining_images_dataset()
    report_path = f'{inference_result_dir}/gender.csv'
    final_score = predict_score_image(remaining_image_loader, cnn_models, report_path)

    print('FINAL PREDICTION SCORE (GENDER) :\n')
    print(final_score)


if __name__ == '__main__':
    main_gender()
