from property_score_cnn import load_cnn_model as load_property_cnn_model

import os
PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# Latent vector z 샘플링 및 해당 z 값으로 생성된 이미지에 대한 semantic score 계산
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator
# - property_score_cnn    (nn.Module) : 핵심 속성 값 계산용 CNN 모델
# - n                     (int)       : sampling 할 latent vector z 의 개수

# Returns:
# - latent_vectors  (NumPy array) : sampling 된 latent vector
# - property_scores (NumPy array) : sampling 된 latent vector 로 생성된 이미지의 핵심 속성값 (eyes, mouth, pose 순서)

def sample_z_and_compute_property_scores(finetune_v1_generator, property_score_cnn, n=10000):
    raise NotImplementedError


# 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지를 각각 추출
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - property_scores (NumPy array) : sampling 된 latent vector 로 생성된 이미지의 핵심 속성값 (eyes, mouth, pose 순서)

# Returns:
# - indices_info (dict) : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 인덱스 정보
#                         {'eyes_largest': list(int), 'eyes_smallest': list(int),
#                          'eyes_largest': list(int), 'eyes_smallest': list(int),
#                          'eyes_largest': list(int), 'eyes_smallest': list(int)}

def extract_best_and_worst_k_images(property_scores, k=1000):
    raise NotImplementedError


# 각 핵심 속성 값 별 핵심 속성 값이 가장 큰 & 작은 k 장의 이미지에 대해 t-SNE 를 이용하여 핵심 속성 값의 시각적 분포 파악
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - latent_vectors (NumPy array) : sampling 된 latent vector
# - indices_info   (dict)        : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 인덱스 정보
#                                  {'eyes_largest': list(int), 'eyes_smallest': list(int),
#                                   'eyes_largest': list(int), 'eyes_smallest': list(int),
#                                   'eyes_largest': list(int), 'eyes_smallest': list(int)}

# Returns:
# - stylegan/stylegan_vectorfind_v6/tsne_result 디렉토리에 각 핵심 속성 값 별 t-SNE 시각화 결과 저장

def run_tsne(latent_vectors, indices_info):
    raise NotImplementedError


# 핵심 속성 값의 변화를 나타내는 latent z vector 를 도출하기 위한 SVM 학습
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - latent_vectors (NumPy array) : sampling 된 latent vector
# - indices_info   (dict)        : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 인덱스 정보
#                                  {'eyes_largest': list(int), 'eyes_smallest': list(int),
#                                   'eyes_largest': list(int), 'eyes_smallest': list(int),
#                                   'eyes_largest': list(int), 'eyes_smallest': list(int)}

# Returns:
# - svm (SVM) : 학습된 SVM (Support Vector Machine)

def train_svm(latent_vectors, indices_info):
    raise NotImplementedError


# SVM 을 이용하여 핵심 속성 값의 변화를 나타내는 latent z vector 를 도출
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - svm                   (SVM)       : 학습된 SVM (Support Vector Machine)
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator
# - property_score_cnn    (nn.Module) : 핵심 속성 값 계산용 CNN 모델

# Returns:
# - property_score_vectors (dict) : 핵심 속성 값의 변화를 나타내는 latent z vector
#                                   {'eyes_vector': NumPy array,
#                                    'mouth_vector': Numpy array,
#                                    'pose_vector': Numpy array}

def find_property_score_vectors(svm, finetune_v1_generator, property_score_cnn):
    raise NotImplementedError


# 핵심 속성 값의 변화를 나타내는 latent z vector 에 대한 정보 저장
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - property_score_vectors (dict) : 핵심 속성 값의 변화를 나타내는 latent z vector
#                                   {'eyes_vector': NumPy array,
#                                    'mouth_vector': Numpy array,
#                                    'pose_vector': Numpy array}

# Returns:
# - stylegan/stylegan_vectorfind_v6/property_score_vectors 디렉토리에 핵심 속성 값의 변화를 나타내는 latent z vector 정보 저장

def save_property_score_vectors_info(property_score_vectors):
    raise NotImplementedError


# StyleGAN-FineTune-v1 모델을 이용한 vector find 실시
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

def run_stylegan_vector_find(finetune_v1_generator, device):
    property_cnn_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_score_cnn = load_property_cnn_model(property_cnn_path, device)

    # latent vector z 샘플링 & 핵심 속성 값이 가장 큰/작은 이미지 추출
    latent_vectors, property_scores = sample_z_and_compute_property_scores(finetune_v1_generator, property_score_cnn)
    indices_info = extract_best_and_worst_k_images(property_scores)
    run_tsne(latent_vectors, indices_info)

    # SVM 학습 & 해당 SVM 으로 핵심 속성 값의 변화를 나타내는 latent z vector 도출
    svm = train_svm(latent_vectors, indices_info)
    property_score_vectors = find_property_score_vectors(svm, finetune_v1_generator, property_score_cnn)

    save_property_score_vectors_info(property_score_vectors)
