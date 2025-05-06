from property_score_cnn import load_cnn_model as load_property_cnn_model
from common import stylegan_transform
import stylegan_common.stylegan_generator_inference as infer

import numpy as np
import torch
import os
import pandas as pd
import plotly.express as px

from sklearn import svm
from sklearn.manifold import TSNE

import sys
global_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
sys.path.append(global_path)

from global_common.visualize_tensor import save_tensor_png

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS_Z = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값
BATCH_SIZE = 20


# Latent vector z 샘플링 및 해당 z 값으로 생성된 이미지에 대한 semantic score 계산
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator
# - property_score_cnn    (nn.Module) : 핵심 속성 값 계산용 CNN 모델
# - n                     (int)       : sampling 할 latent vector z 의 개수

# Returns:
# - latent_vectors  (NumPy array) : sampling 된 latent vector
# - property_scores (dict)        : sampling 된 latent vector 로 생성된 이미지의 (Pre-trained CNN 에 의해 도출된) 핵심 속성값
#                                   {'eyes_cnn_score': list(float),
#                                    'mouth_cnn_score': list(float),
#                                    'pose_cnn_score': list(float)}

def sample_z_and_compute_property_scores(finetune_v1_generator, property_score_cnn, n=5000):
    save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/inference_test_during_training'

    z = np.random.normal(0, 1, size=(n, ORIGINAL_HIDDEN_DIMS_Z)).astype(np.float64)
    additional = np.random.normal(0, 1, size=(n, ORIGINALLY_PROPERTY_DIMS_Z)).astype(np.float64)
    latent_vectors = np.concatenate([z, additional], axis=1)

    eyes_cnn_scores = []
    mouth_cnn_scores = []
    pose_cnn_scores = []

    for i in range(n // BATCH_SIZE):
        if i % 10 == 0:
            print(f'synthesizing for batch {i} ...')

        z_ = z[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
        additional_ = additional[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

        images = infer.synthesize(finetune_v1_generator,
                                  num=BATCH_SIZE,
                                  save_dir=save_dir,
                                  z=z_,
                                  label=additional_,
                                  img_name_start_idx=0,
                                  verbose=False,
                                  save_img=False,
                                  return_img=True)

        with torch.no_grad():
            for image_no in range(BATCH_SIZE):
                image = images[image_no]
                image_ = image / 255.0
                image_ = (image_ - 0.5) / 0.5
                image_ = torch.tensor(image_).type(torch.float32)
                image_ = image_.permute(2, 0, 1)

                property_scores = property_score_cnn(image_.unsqueeze(0).cuda())
                property_score_np = property_scores.detach().cpu().numpy()

                eyes_cnn_scores.append(property_score_np[0][0])
                mouth_cnn_scores.append(property_score_np[0][3])
                pose_cnn_scores.append(property_score_np[0][4])

    property_scores = {'eyes_cnn_score': eyes_cnn_scores,
                       'mouth_cnn_score': mouth_cnn_scores,
                       'pose_cnn_score': pose_cnn_scores}

    return latent_vectors, property_scores


# 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지를 각각 추출
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - property_scores (dict) : sampling 된 latent vector 로 생성된 이미지의 (Pre-trained CNN 에 의해 도출된) 핵심 속성값
#                            {'eyes_cnn_score': list(float),
#                             'mouth_cnn_score': list(float),
#                             'pose_cnn_score': list(float)}

# Returns:
# - indices_info (dict) : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 인덱스 정보
#                         {'eyes_largest': list(int), 'eyes_smallest': list(int),
#                          'mouth_largest': list(int), 'mouth_smallest': list(int),
#                          'pose_largest': list(int), 'pose_smallest': list(int)}

def extract_best_and_worst_k_images(property_scores, k=50):

    # sort scores with index
    eyes_cnn_scores_with_idx = []
    for i in range(len(property_scores['eyes_cnn_score'])):
        eyes_cnn_scores_with_idx.append([i, property_scores['eyes_cnn_score'][i]])

    mouth_cnn_scores_with_idx = []
    for i in range(len(property_scores['mouth_cnn_score'])):
        mouth_cnn_scores_with_idx.append([i, property_scores['mouth_cnn_score'][i]])

    pose_cnn_scores_with_idx = []
    for i in range(len(property_scores['pose_cnn_score'])):
        pose_cnn_scores_with_idx.append([i, property_scores['pose_cnn_score'][i]])

    eyes_cnn_scores_with_idx.sort(key=lambda x: x[1], reverse=True)
    mouth_cnn_scores_with_idx.sort(key=lambda x: x[1], reverse=True)
    pose_cnn_scores_with_idx.sort(key=lambda x: x[1], reverse=True)

    # generate largest/smallest score index info
    eyes_largest_idxs = sorted([x[0] for x in eyes_cnn_scores_with_idx][:k])
    mouth_largest_idxs = sorted([x[0] for x in mouth_cnn_scores_with_idx][:k])
    pose_largest_idxs = sorted([x[0] for x in pose_cnn_scores_with_idx][:k])

    eyes_smallest_idxs = sorted([x[0] for x in eyes_cnn_scores_with_idx][-k:])
    mouth_smallest_idxs = sorted([x[0] for x in mouth_cnn_scores_with_idx][-k:])
    pose_smallest_idxs = sorted([x[0] for x in pose_cnn_scores_with_idx][-k:])

    indices_info = {
        'eyes_largest': eyes_largest_idxs, 'eyes_smallest': eyes_smallest_idxs,
        'mouth_largest': mouth_largest_idxs, 'mouth_smallest': mouth_smallest_idxs,
        'pose_largest': pose_largest_idxs, 'pose_smallest': pose_smallest_idxs
    }

    return indices_info


# 각 핵심 속성 값 별 핵심 속성 값이 가장 큰 & 작은 k 장의 이미지에 대해 t-SNE 를 이용하여 핵심 속성 값의 시각적 분포 파악
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - latent_vectors (NumPy array) : sampling 된 latent vector
# - indices_info   (dict)        : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 인덱스 정보
#                                  {'eyes_largest': list(int), 'eyes_smallest': list(int),
#                                   'mouth_largest': list(int), 'mouth_smallest': list(int),
#                                   'pose_largest': list(int), 'pose_smallest': list(int)}

# Returns:
# - stylegan/stylegan_vectorfind_v6/tsne_result 디렉토리에 각 핵심 속성 값 별 t-SNE 시각화 결과 저장

def run_tsne(latent_vectors, indices_info):
    property_names = ['eyes', 'mouth', 'pose']
    tsne_result_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v6/tsne_result'
    os.makedirs(tsne_result_path, exist_ok=True)

    for property_name in property_names:
        largest_img_idxs = indices_info[f'{property_name}_largest']
        smallest_img_idxs = indices_info[f'{property_name}_smallest']
        idxs = largest_img_idxs + smallest_img_idxs
        indexed_latent_vectors = latent_vectors[idxs]

        # run t-SNE
        print(f'running t-SNE for {property_name} ...')
        tsne_result = TSNE(n_components=2,
                           perplexity=25,
                           learning_rate=100,
                           n_iter=500,
                           random_state=2025).fit_transform(indexed_latent_vectors)

        # save t-SNE plot result images
        classes = ['largest'] * len(largest_img_idxs) + ['smallest'] * len(smallest_img_idxs)

        data_dict = {
            'dimension_0': tsne_result[:, 0],
            'dimension_1': tsne_result[:, 1],
            'class': classes
        }
        data_df = pd.DataFrame(data_dict)

        fig = px.scatter(data_df,
                         x='dimension_0',
                         y='dimension_1',
                         color='class',
                         title=f't-SNE result of property {property_name}')

        fig.update_layout(width=900, height=750)
        fig.update_traces(marker=dict(size=5))
        fig.write_image(f'{tsne_result_path}/tsne_result_{property_name}.png')


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
# - svm_classifiers (dict(SVM)) : 학습된 SVM (Support Vector Machine)
#                                 {'eyes': SVM, 'mouth': SVM, 'pose': SVM}

def train_svm(latent_vectors, indices_info):
    property_names = ['eyes', 'mouth', 'pose']
    train_ratio = 0.8
    svm_classifiers = {}

    # use option from original paper (higan/blob/master/utils/boundary_searcher.py GenForce GitHub)
    for property_name in property_names:

        # create dataset
        largest_img_idxs = indices_info[f'{property_name}_largest']
        smallest_img_idxs = indices_info[f'{property_name}_smallest']
        largest_train_count = int(train_ratio * len(largest_img_idxs))
        smallest_train_count = int(train_ratio * len(smallest_img_idxs))

        train_idxs = largest_img_idxs[:largest_train_count] + smallest_img_idxs[:smallest_train_count]
        valid_idxs = largest_img_idxs[largest_train_count:] + smallest_img_idxs[smallest_train_count:]
        train_latent_vectors = latent_vectors[train_idxs]
        valid_latent_vectors = latent_vectors[valid_idxs]

        train_classes = ['largest'] * largest_train_count + ['smallest'] * smallest_train_count
        valid_classes = ['largest'] * (len(largest_img_idxs) - largest_train_count) + ['smallest'] * (len(smallest_img_idxs) - smallest_train_count)

        # train SVM
        print(f'\ntraining SVM for {property_name} ...')
        svm_clf = svm.SVC(kernel='linear')
        svm_classifier = svm_clf.fit(train_latent_vectors, train_classes)

        # valid SVM
        valid_predictions = svm_classifier.predict(valid_latent_vectors)

        # compute performance metric
        large_large = np.sum((np.array(valid_predictions) == 'largest') & (np.array(valid_classes) == 'largest'))
        large_small = np.sum((np.array(valid_predictions) == 'largest') & (np.array(valid_classes) == 'smallest'))
        small_large = np.sum((np.array(valid_predictions) == 'smallest') & (np.array(valid_classes) == 'largest'))
        small_small = np.sum((np.array(valid_predictions) == 'smallest') & (np.array(valid_classes) == 'smallest'))

        accuracy = (large_large + small_small) / (large_large + large_small + small_large + small_small)

        large_recall = large_large / (large_large + small_large)
        large_precision = large_large / (large_large + large_small)
        large_f1 = 2 * large_recall * large_precision / (large_recall + large_precision)

        small_recall = small_small / (small_small + large_small)
        small_precision = small_small / (small_small + small_large)
        small_f1 = 2 * small_recall * small_precision / (small_recall + small_precision)

        print(f'accuracy          : {accuracy:.4f}')
        print(f'recall    (large) : {large_recall:.4f}')
        print(f'precision (large) : {large_precision:.4f}')
        print(f'F1 score  (large) : {large_f1:.4f}')
        print(f'recall    (small) : {small_recall:.4f}')
        print(f'precision (small) : {small_precision:.4f}')
        print(f'F1 score  (small) : {small_f1:.4f}')

        svm_classifiers[property_name] = svm_classifier

    return svm_classifiers


# SVM 을 이용하여 핵심 속성 값의 변화를 나타내는 latent z vector 를 도출
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - svm_classifiers       (dict(SVM)) : 학습된 SVM (Support Vector Machine)
#                                       {'eyes': SVM, 'mouth': SVM, 'pose': SVM}
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator
# - property_score_cnn    (nn.Module) : 핵심 속성 값 계산용 CNN 모델

# Returns:
# - property_score_vectors (dict) : 핵심 속성 값의 변화를 나타내는 latent z vector
#                                   {'eyes_vector': NumPy array,
#                                    'mouth_vector': Numpy array,
#                                    'pose_vector': Numpy array}

def find_property_score_vectors(svm_classifiers, finetune_v1_generator, property_score_cnn):
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
    svm_classifiers = train_svm(latent_vectors, indices_info)
    property_score_vectors = find_property_score_vectors(svm_classifiers, finetune_v1_generator, property_score_cnn)

    save_property_score_vectors_info(property_score_vectors)
