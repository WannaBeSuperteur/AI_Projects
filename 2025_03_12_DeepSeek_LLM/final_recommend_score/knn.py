import os

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
# - test_diagrams      (PyTorch Tensor) : 테스트 대상 다이어그램을 128 x 128 + 어둡게 변환 후 한번에 로딩한 PyTorch Tensor
#                                         shape : (N, 3, 128, 128)
# - test_diagram_paths (list(str))      : test diagram 의 경로 리스트

def load_test_diagrams(test_diagram_dir=TEST_DIAGRAM_PATH):
    diagram_img_names = os.listdir(test_diagram_dir)
    diagram_img_names = list(filter(lambda x: x.startswith('test_diagram') and x.endswith('.png'), diagram_img_names))
    N = len(diagram_img_names)

    test_diagrams = torch.zeros((N, 3, IMG_HEIGHT, IMG_WIDTH))
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
# - scored_diagram_paths (dict) : 각 점수별 다이어그램 파일 경로의 목록
#                                 {0: list(str), 1: list(str), 2: list(str), 3: list(str), 4: list(str), 5: list(str)}

def load_user_score_data():
    scored_diagram_paths = {}

    for score in [0, 1, 2, 3, 4, 5]:
        diagram_dir = f'{KNN_TRAIN_DATASET_PATH}/{score}'

        if os.path.exists(diagram_dir):
            scored_diagram_paths[score] = list(os.listdir(diagram_dir))
            scored_diagram_paths[score] = list(filter(lambda x: x.endswith('png'), scored_diagram_paths[score]))

        else:
            scored_diagram_paths[score] = []

    return scored_diagram_paths


# 각 테스트 대상 다이어그램의 latent vector 와 사용자 평가가 이루어진 각 다이어그램의 latent vector 간의 거리 계산
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - ae_encoder           (nn.Module)      : Auto-Encoder 의 Encoder
# - test_diagrams        (PyTorch Tensor) : 테스트 대상 다이어그램 T 개를 128 x 128 + 어둡게 변환 후 한번에 로딩한 PyTorch Tensor
#                                           shape : (T, 3, 128, 128)
# - scored_diagram_paths (dict)           : 각 점수별 다이어그램 파일 경로의 목록
#                                           {0: list(str), 1: list(str), 2: list(str), ..., 5: list(str)}

# Returns:
# - distance_df (Pandas DataFrame) : latent vector 간의 거리 계산 결과
#                                    - row 는 사용자 평가가 이루어진 각 다이어그램을 나타냄
#                                    - column 은 이미지 경로 (1개) + 점수 (1개) + 각 테스트 다이어그램 (T개) = 총 2 + T 개

def compute_distance(ae_encoder, test_diagrams, scored_diagram_paths):
    raise NotImplementedError


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
    scored_diagram_paths = load_user_score_data()

    # 거리 및 예상 사용자 평가 점수 계산
    distance_df = compute_distance(ae_encoder, test_diagrams, scored_diagram_paths)
    final_score = compute_final_score(distance_df)

    print('FINAL SCORE :')
    print(final_score)
