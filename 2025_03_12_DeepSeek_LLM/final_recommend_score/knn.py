
# Auto-Encoder 모델 로딩
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - ae_encoder (nn.Module) : Auto-Encoder 의 Encoder

def load_ae_model():
    raise NotImplementedError


# diagrams_for_test/test_diagram_{i}.png 의 테스트 대상 다이어그램 로딩
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - test_diagrams (PyTorch Tensor) : 테스트 대상 다이어그램을 128 x 128 + 어둡게 변환 후 한번에 로딩한 PyTorch Tensor
#                                    shape : (N, 3, 128, 128)

def load_test_diagrams():
    raise NotImplementedError


# knn_user_score/{0,1,2,3,4,5} 안에 있는, 사용자에 의해 점수가 매겨진 다이어그램 로딩
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - scored_diagram_paths (dict) : 각 점수별 다이어그램 파일 경로의 목록
#                                 {0: list(str), 1: list(str), 2: list(str), 3: list(str), 4: list(str), 5: list(str)}

def load_user_score_data():
    raise NotImplementedError


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
    ae_encoder = load_ae_model()
    test_diagrams = load_test_diagrams()
    scored_diagram_paths = load_user_score_data()

    # 거리 및 예상 사용자 평가 점수 계산
    distance_df = compute_distance(ae_encoder, test_diagrams, scored_diagram_paths)
    final_score = compute_final_score(distance_df)

    print('FINAL SCORE :')
    print(final_score)
