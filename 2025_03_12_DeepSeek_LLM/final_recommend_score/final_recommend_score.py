try:
    from knn import load_test_diagrams, load_user_score_data, compute_distance, compute_final_score
    from ae import load_ae_encoder
    from cnn import load_cnn_model
    from common import IMG_HEIGHT, IMG_WIDTH
except:
    from final_recommend_score.knn import load_test_diagrams, load_user_score_data, compute_distance, compute_final_score
    from final_recommend_score.ae import load_ae_encoder
    from final_recommend_score.cnn import load_cnn_model
    from final_recommend_score.common import IMG_HEIGHT, IMG_WIDTH

import numpy as np
import pandas as pd

import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TEST_DIAGRAM_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/diagrams_for_test'


# Auto-Encoder 의 Encoder 및 CNN 모델 로딩
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - cnn_models (list(nn.Module)) : CNN 모델 리스트
# - ae_encoder (nn.Module)       : Auto-Encoder 의 인코더

def load_models():
    cnn_models = load_cnn_model()
    ae_encoder = load_ae_encoder()
    print('model load successful!')

    return cnn_models, ae_encoder


# CNN 을 이용한 "기본 가독성 점수" 계산
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - cnn_models         (list(nn.Module)) : CNN 모델 리스트
# - test_diagrams      (PyTorch Tensor)  : 테스트 대상 다이어그램 T 개를 128 x 128 + 어둡게 변환 후 한번에 로딩한 PyTorch Tensor
#                                          shape : (T, 3, 128, 128)
# - test_diagram_paths (list(str))       : test diagram 의 경로 리스트

# Returns:
# - final_cnn_score_df (Pandas DataFrame) : 각 테스트 다이어그램에 대한 기본 가독성 점수 계산 결과
#                                           - columns = ['img_path', 'final_score']

def compute_cnn_score(cnn_models, test_diagrams, test_diagram_paths):
    final_cnn_score_dict = {'img_path': test_diagram_paths, 'final_score': []}

    for test_diagram, test_diagram_path in zip(test_diagrams, test_diagram_paths):
        scores = []
        test_diagram_for_cnn = test_diagram.reshape((1, 3, IMG_HEIGHT, IMG_WIDTH))

        for cnn_model in cnn_models:
            cnn_output = np.array(cnn_model(test_diagram_for_cnn)[0].detach().cpu())
            cnn_score = 5.0 * cnn_output
            scores.append(cnn_score)

        mean_cnn_score = np.mean(scores)
        final_cnn_score_dict['final_score'].append(mean_cnn_score)

    final_cnn_score_df = pd.DataFrame(final_cnn_score_dict)
    final_cnn_score_df['img_path'] = final_cnn_score_df['img_path'].apply(lambda x: x.split(f'{PROJECT_DIR_PATH}/')[1])

    return final_cnn_score_df


# Auto-Encoder 의 Encoder 를 이용한 "예상 사용자 평가 점수" 계산
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - ae_encoder         (nn.Module)      : Auto-Encoder 의 인코더
# - test_diagrams      (PyTorch Tensor) : 테스트 대상 다이어그램 T 개를 128 x 128 + 어둡게 변환 후 한번에 로딩한 PyTorch Tensor
#                                         shape : (T, 3, 128, 128)
# - test_diagram_paths (list(str))      : test diagram 의 경로 리스트

# Returns:
# - final_ae_score_df (Pandas DataFrame) : 각 테스트 다이어그램에 대한 예상 사용자 평가 점수 계산 결과
#                                          - columns = ['img_path', 'final_score']

def compute_ae_encoder_score(ae_encoder, test_diagrams, test_diagram_paths):
    scored_diagram_paths_dict = load_user_score_data()

    # 거리 및 예상 사용자 평가 점수 계산
    distance_df = compute_distance(ae_encoder, test_diagrams, test_diagram_paths, scored_diagram_paths_dict, save_df=False)
    final_ae_score_df = compute_final_score(distance_df, save_df=False)

    return final_ae_score_df


# "최종 점수" = "기본 가독성 점수" (CNN) + "예상 사용자 평가 점수" (Auto-Encoder) 계산
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - cnn_score_df (Pandas DataFrame) : 각 테스트 다이어그램에 대한 기본 가독성 점수 계산 결과
#                                     - columns = ['img_path', 'final_score']
# - ae_score_df  (Pandas DataFrame) : 각 테스트 다이어그램에 대한 예상 사용자 평가 점수 계산 결과
#                                     - columns = ['img_path', 'final_score']
# - save_df      (bool)             : final_score_df 를 log/final_recommend_result.csv 로 저장할지 여부

# Returns:
# - final_recommend_score_df (Pandas DataFrame) : 각 테스트 다이어그램에 대한 "최종 점수" = 기본 가독성 + 예상 사용자 평가 계산 결과
#                                                 column: ['img_path', 'cnn_read_score', 'ae_user_score', 'final_score']
# - save_df = True 이면 final_recommend_score_df 를 log/final_recommend_result.csv 로 저장

def compute_final_recommend_score(cnn_score_df, ae_score_df, save_df=True):

    final_recommend_score_dict = {
        'img_path': cnn_score_df['img_path'],
        'cnn_read_score': cnn_score_df['final_score'],
        'ae_user_score': ae_score_df['final_score']
    }
    final_recommend_score_df = pd.DataFrame(final_recommend_score_dict)

    final_recommend_score_df['final_score'] = final_recommend_score_df.apply(
        lambda x: x['cnn_read_score'] + x['ae_user_score'],
        axis=1)

    if save_df:
        final_recommend_scores_csv_path = f'{PROJECT_DIR_PATH}/final_recommend_score/log/final_recommend_result.csv'
        final_recommend_score_df.to_csv(final_recommend_scores_csv_path, index=False)

    return final_recommend_score_df


if __name__ == '__main__':
    cnn_models, ae_encoder = load_models()
    test_diagrams, test_diagram_paths = load_test_diagrams(test_diagram_dir=TEST_DIAGRAM_PATH)

    cnn_score_df = compute_cnn_score(cnn_models, test_diagrams, test_diagram_paths)
    ae_score_df = compute_ae_encoder_score(ae_encoder, test_diagrams, test_diagram_paths)

    final_recommend_score_df = compute_final_recommend_score(cnn_score_df, ae_score_df)

    print('FINAL RECOMMEND SCORE :')
    print(final_recommend_score_df)
