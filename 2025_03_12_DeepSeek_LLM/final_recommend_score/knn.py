import os
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms

from ae import load_ae_encoder
from common import resize_and_darken_image, IMG_HEIGHT, IMG_WIDTH

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TEST_DIAGRAM_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/diagrams_for_test'
KNN_TRAIN_DATASET_PATH = f'{PROJECT_DIR_PATH}/final_recommend_score/knn_user_score'


# diagrams_for_test/test_diagram_{i}.png 의 테스트 대상 다이어그램 로딩
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - test_diagram_dir (str) : test diagram 이 있는 디렉토리 경로 (기본적으로 diagrams_for_test 를 가리킴)

# Returns:
# - test_diagrams      (PyTorch Tensor) : 테스트 대상 다이어그램 T 개를 128 x 128 + 어둡게 변환 후 한번에 로딩한 PyTorch Tensor
#                                         shape : (T, 3, 128, 128)
# - test_diagram_paths (list(str))      : test diagram 의 경로 리스트

def load_test_diagrams(test_diagram_dir=TEST_DIAGRAM_PATH):
    diagram_img_names = os.listdir(test_diagram_dir)
    diagram_img_names = list(filter(lambda x: x.startswith('test_diagram') and x.endswith('.png'), diagram_img_names))
    T = len(diagram_img_names)

    test_diagrams = torch.zeros((T, 3, IMG_HEIGHT, IMG_WIDTH))
    test_diagram_paths = []

    for idx, diagram_img_name in enumerate(diagram_img_names):
        img_full_path = f'{test_diagram_dir}/{diagram_img_name}'
        test_diagram_paths.append(img_full_path)

        img = resize_and_darken_image(img_full_path, dest_width=IMG_WIDTH, dest_height=IMG_HEIGHT)
        img_tensor = transforms.ToTensor()(img) / 255.0
        test_diagrams[idx] = img_tensor.reshape((3, IMG_HEIGHT, IMG_WIDTH))

    return test_diagrams, test_diagram_paths


# knn_user_score/{0,1,2,3,4,5} 안에 있는, 사용자에 의해 점수가 매겨진 다이어그램 로딩
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - scored_diagram_paths_dict (dict) : 각 점수별 다이어그램 파일 경로의 목록
#                                      {0: list(str), 1: list(str), 2: list(str), ..., 5: list(str)}

def load_user_score_data():
    scored_diagram_paths_dict = {}

    for score in [0, 1, 2, 3, 4, 5]:
        diagram_dir = f'{KNN_TRAIN_DATASET_PATH}/{score}'

        if os.path.exists(diagram_dir):
            scored_diagram_paths_dict[score] = list(os.listdir(diagram_dir))
            scored_diagram_paths_dict[score] = list(filter(lambda x: x.endswith('png'),
                                                           scored_diagram_paths_dict[score]))
            scored_diagram_paths_dict[score] = list(map(lambda x: f'{diagram_dir}/{x}',
                                                        scored_diagram_paths_dict[score]))

        else:
            scored_diagram_paths_dict[score] = []

    return scored_diagram_paths_dict


# 각 테스트 대상 다이어그램의 latent vector 와 사용자 평가가 이루어진 각 다이어그램의 latent vector 간의 거리 계산
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - ae_encoder                (nn.Module)      : Auto-Encoder 의 Encoder
# - test_diagrams             (PyTorch Tensor) : 테스트 대상 다이어그램 T 개를 128 x 128 + 어둡게 변환 후 한번에 로딩한 PyTorch Tensor
#                                                shape : (T, 3, 128, 128)
# - test_diagram_paths        (list(str))      : test diagram 의 경로 리스트
# - scored_diagram_paths_dict (dict)           : 각 점수별 다이어그램 파일 경로의 목록 (총 이미지 S 개)
#                                                {0: list(str), 1: list(str), 2: list(str), ..., 5: list(str)}

# Returns:
# - distance_df (Pandas DataFrame) : latent vector 간의 거리 계산 결과
#                                    - row 는 사용자 평가가 이루어진 각 다이어그램을 나타냄
#                                    - column 은 이미지 경로 (1개) + 점수 (1개) + 각 테스트 다이어그램 (T개) = 총 2 + T 개
# - distance_df 를 log/knn_distances.csv 로 저장

def compute_distance(ae_encoder, test_diagrams, test_diagram_paths, scored_diagram_paths_dict):
    scores = [0, 1, 2, 3, 4, 5]

    # diagram path & score dict
    path_and_score_dict = {'img_path': [], 'score': []}
    for score in scores:
        for path in scored_diagram_paths_dict[score]:
            path_and_score_dict['img_path'].append(path)
            path_and_score_dict['score'].append(score)

    # load scored diagrams as PyTorch tensor
    S = sum(len(scored_diagram_paths_dict[score]) for score in scores)
    T = len(test_diagram_paths)

    scored_diagrams = torch.zeros((S, 3, IMG_HEIGHT, IMG_WIDTH))
    scored_diagram_paths = path_and_score_dict['img_path']

    for idx, scored_diagram_path in enumerate(scored_diagram_paths):
        img = resize_and_darken_image(scored_diagram_path, dest_width=IMG_WIDTH, dest_height=IMG_HEIGHT)
        img_tensor = transforms.ToTensor()(img) / 255.0
        scored_diagrams[idx] = img_tensor.reshape((3, IMG_HEIGHT, IMG_WIDTH))

    # compute latent vector from test diagrams and scored diagrams
    distance_arr = np.zeros((S, T))

    latent_vector_scored_diagrams = []
    latent_vector_test_diagrams = []

    for s in range(S):
        scored_diagram = scored_diagrams[s].reshape(1, 3, IMG_HEIGHT, IMG_WIDTH)
        scored_diagram = scored_diagram[:, :, IMG_HEIGHT // 4: 3 * IMG_HEIGHT // 4, IMG_WIDTH // 4: 3 * IMG_WIDTH // 4]

        latent_vector_scored_diagram = np.array(ae_encoder(scored_diagram).detach().cpu())
        latent_vector_scored_diagrams.append(latent_vector_scored_diagram)

    for t in range(T):
        test_diagram = test_diagrams[t].reshape(1, 3, IMG_HEIGHT, IMG_WIDTH)
        test_diagram = test_diagram[:, :, IMG_HEIGHT // 4: 3 * IMG_HEIGHT // 4, IMG_WIDTH // 4: 3 * IMG_WIDTH // 4]

        latent_vector_test_diagram = np.array(ae_encoder(test_diagram).detach().cpu())
        latent_vector_test_diagrams.append(latent_vector_test_diagram)

    for s in range(S):  # for each scored diagram
        for t in range(T):  # for each test diagram
            latent_vector_scored = latent_vector_scored_diagrams[s]
            latent_vector_test = latent_vector_test_diagrams[t]

            distance = np.sum(np.square(latent_vector_scored - latent_vector_test))
            distance_arr[s][t] = distance

    # create and return distance DataFrame
    distance_df = pd.DataFrame(path_and_score_dict)
    distance_arr = np.round(distance_arr, 4)
    distance_arr_df = pd.DataFrame(distance_arr)
    distance_arr_df.columns = list(map(lambda x: x.split(f'{PROJECT_DIR_PATH}/')[1], test_diagram_paths))

    distance_df = pd.concat([distance_df, distance_arr_df], axis=1)
    distance_df['img_path'] = distance_df['img_path'].apply(lambda x: x.split('knn_user_score/')[1])

    knn_distance_csv_path = f'{PROJECT_DIR_PATH}/final_recommend_score/log/knn_distances.csv'
    distance_df.to_csv(knn_distance_csv_path, index=False)

    return distance_df


# 각 다이어그램의 latent vector 간의 거리 정보를 이용하여 각 테스트 다이어그램의 최종 점수 산출
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - distance_df (Pandas DataFrame) : latent vector 간의 거리 계산 결과
#                                    - row 는 사용자 평가가 이루어진 각 다이어그램을 나타냄
#                                    - column 은 이미지 경로 (1개) + 점수 (1개) + 각 테스트 다이어그램 (T개) = 총 2 + T 개

# Returns:
# - final_score_df (Pandas DataFrame) : 각 테스트 다이어그램에 대한 최종 점수 (예상 사용자 평가 점수) 계산 결과
#                                       - columns = ['img_path', 'final_score']

def compute_final_score(distance_df):
    raise NotImplementedError


if __name__ == '__main__':

    # 모델 및 다이어그램 정보 로딩
    ae_encoder = load_ae_encoder()
    test_diagrams, test_diagram_paths = load_test_diagrams()
    scored_diagram_paths_dict = load_user_score_data()

    # 거리 및 예상 사용자 평가 점수 계산
    distance_df = compute_distance(ae_encoder, test_diagrams, test_diagram_paths, scored_diagram_paths_dict)
    final_score = compute_final_score(distance_df)

    print('FINAL SCORE :')
    print(final_score)
