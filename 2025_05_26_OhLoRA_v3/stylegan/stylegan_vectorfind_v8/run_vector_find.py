import numpy as np
import torch
import pandas as pd
import plotly.express as px

from sklearn import svm
from sklearn.manifold import TSNE

import random
import time
import os
import sys

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(PROJECT_DIR_PATH)

try:
    import stylegan_common.stylegan_generator_inference as infer
    from common import load_merged_property_score_cnn
except:
    import stylegan.stylegan_common.stylegan_generator_inference as infer
    from stylegan.common import load_merged_property_score_cnn

# use sklearnex (scikit-learn-intelex) library for speedup SVM training
from sklearnex import patch_sklearn
patch_sklearn()

# remove warnings
import warnings
warnings.filterwarnings('ignore')


ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINAL_HIDDEN_DIMS_W = 512
ORIGINALLY_PROPERTY_DIMS = 7    # 원래 property (eyes, hair_color, hair_length, mouth, pose,
                                #               background_mean, background_std) 목적으로 사용된 dimension 값
BATCH_SIZE = 20
SVMS_PER_EACH_PROPERTY = 1      # also w-vector count for each property

GROUP_NAMES = ['hhhh', 'hhhl', 'hhlh', 'hhll', 'hlhh', 'hlhl', 'hllh', 'hlll',
               'lhhh', 'lhhl', 'lhlh', 'lhll', 'llhh', 'llhl', 'lllh', 'llll']
PROPERTY_NAMES = ['eyes', 'mouth', 'pose']


# intermediate w vector 로 생성된 이미지를 머리 색, 머리 길이, 배경 색 평균, "직모 vs. 곱슬 (hairstyle)" 에 따라 그룹화하기 위해,
# hair_color, hair_length, background_mean, hairstyle 핵심 속성 값의 중앙값 얻기

# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - medians (dict(float)) : hair_color, hair_length, background_mean, hairstyle 핵심 속성 값의 중앙값
#                           {'hair_color': float, 'hair_length': float, 'background_mean': float, 'hairstyle': float}

def get_medians():
    mean_and_median_csv_path = f'{PROJECT_DIR_PATH}/v8_property_scores/property_scores_mean_and_median.csv'
    mean_and_median_df = pd.read_csv(mean_and_median_csv_path)

    hair_color_median = mean_and_median_df['hair_color'][1]
    hair_length_median = mean_and_median_df['hair_length'][1]
    background_mean_median = mean_and_median_df['background_score'][1]
    hairstyle_median = mean_and_median_df['hairstyle'][1]

    medians = {'hair_color': hair_color_median,
               'hair_length': hair_length_median,
               'background_mean': background_mean_median,
               'hairstyle': hairstyle_median}

    print(f'medians = {medians}')

    return medians


# intermediate w vector 샘플링 및 해당 w 값으로 생성된 이미지에 대한 semantic score 계산
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - finetune_v8_generator (nn.Module) : StyleGAN-FineTune-v8 의 Generator
# - property_score_cnn    (nn.Module) : 핵심 속성 값 계산용 CNN 모델
# - n                     (int)       : sampling 할 intermediate w vector 의 개수

# Returns:
# - w_vectors_by_group (dict(NumPy array)) : sampling 된 intermediate w (각 그룹별)
# - property_scores    (dict)              : sampling 된 intermediate w 로 생성된 이미지의 Pre-trained CNN 도출 핵심 속성값
#                                            dict 는 각 그룹의 이름 ('hhhh', 'hhhl', ..., 'llll') 을 key 로 함
#                                            {'eyes_cnn_score': dict(list(float)),
#                                             'mouth_cnn_score': dict(list(float)),
#                                             'pose_cnn_score': dict(list(float))}

def sample_w_and_compute_property_scores(finetune_v8_generator, property_score_cnn, n=20000):
    save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/inference_test_during_training'
    medians = get_medians()  # returned values : -0.4574, 0.5734, 0.7618, -0.0167

    z = np.random.normal(0, 1, size=(n, ORIGINAL_HIDDEN_DIMS_Z)).astype(np.float64)
    w = np.zeros((n, ORIGINAL_HIDDEN_DIMS_Z)).astype(np.float64)
    additional = np.random.normal(0, 1, size=(n, ORIGINALLY_PROPERTY_DIMS)).astype(np.float64)

    # 생성된 이미지를 머리 색, 머리 길이, 배경 색 평균, "직모 vs. 곱슬" 평균의 CNN 도출 속성값에 따라 MBTI 처럼 16 개의 그룹으로 나눔
    # (그룹명 : 머리 색, 머리 길이, 배경 색 평균, "직모 vs. 곱슬" 값 순서로, h: median 보다 높음 / l: median 보다 낮음)
    w_vectors_by_group = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                          'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                          'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                          'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}

    eyes_cnn_scores = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                       'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                       'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                       'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}

    mouth_cnn_scores = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                        'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                        'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                        'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}

    pose_cnn_scores = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                       'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                       'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                       'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}

    for i in range(n // BATCH_SIZE):
        if i % 10 == 0:
            print(f'synthesizing for batch {i} ...')

        z_ = z[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
        additional_ = additional[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

        images, ws = infer.synthesize(finetune_v8_generator,
                                      num=BATCH_SIZE,
                                      save_dir=save_dir,
                                      z=z_,
                                      label=additional_,
                                      img_name_start_idx=0,
                                      verbose=False, save_img=False, return_img=True, return_w=True)

        w[i * BATCH_SIZE : (i+1) * BATCH_SIZE] = ws

        with torch.no_grad():
            for image_no in range(BATCH_SIZE):
                image = images[image_no]
                image_ = image / 255.0
                image_ = (image_ - 0.5) / 0.5
                image_ = torch.tensor(image_).type(torch.float32)
                image_ = image_.permute(2, 0, 1)

                property_scores = property_score_cnn(image_.unsqueeze(0).cuda())
                property_score_np = property_scores.detach().cpu().numpy()

                hair_color_group = 'h' if property_score_np[0][1] >= medians['hair_color'] else 'l'
                hair_length_group = 'h' if property_score_np[0][2] >= medians['hair_length'] else 'l'
                background_mean_group = 'h' if property_score_np[0][5] >= medians['background_mean'] else 'l'
                hairstyle_group = 'h' if property_score_np[0][7] >= medians['hairstyle'] else 'l'
                group_name = hair_color_group + hair_length_group + background_mean_group + hairstyle_group

                eyes_cnn_scores[group_name].append(property_score_np[0][0])
                mouth_cnn_scores[group_name].append(property_score_np[0][3])
                pose_cnn_scores[group_name].append(property_score_np[0][4])

                w_vector = w[i * BATCH_SIZE + image_no]
                w_vectors_by_group[group_name].append(list(w_vector))

    property_scores = {'eyes_cnn_score': eyes_cnn_scores,
                       'mouth_cnn_score': mouth_cnn_scores,
                       'pose_cnn_score': pose_cnn_scores}

    for group_name in GROUP_NAMES:
        w_vectors_by_group[group_name] = np.array(w_vectors_by_group[group_name])
        print(f'generated images in group {group_name} : {len(w_vectors_by_group[group_name])}')

    return w_vectors_by_group, property_scores


# 각 핵심 속성 값이 가장 큰 & 가장 작은 ratio 비율만큼의 이미지를 그룹별로 각각 추출
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - property_scores (dict) : sampling 된 intermediate w vector 로 생성된 이미지의 Pre-trained CNN 도출 핵심 속성값
#                            dict 는 각 그룹의 이름 ('hhhh', 'hhhl', ..., 'llll') 을 key 로 함
#                            {'eyes_cnn_score': dict(list(float)),
#                             'mouth_cnn_score': dict(list(float)),
#                             'pose_cnn_score': dict(list(float))}

# Returns:
# - indices_info (dict) : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 (그룹별) 인덱스 정보
#                         {'eyes_largest': dict(list(int)), 'eyes_smallest': dict(list(int)),
#                          'mouth_largest': dict(list(int)), 'mouth_smallest': dict(list(int)),
#                          'pose_largest': dict(list(int)), 'pose_smallest': dict(list(int))}

def extract_best_and_worst_k_images(property_scores, ratio=0.2):

    eyes_largest_idxs = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                         'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                         'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                         'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}
    mouth_largest_idxs = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                          'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                          'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                          'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}
    pose_largest_idxs = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                         'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                         'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                         'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}

    eyes_smallest_idxs = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                          'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                          'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                          'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}
    mouth_smallest_idxs = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                           'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                           'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                           'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}
    pose_smallest_idxs = {'hhhh': [], 'hhhl': [], 'hhlh': [], 'hhll': [],
                          'hlhh': [], 'hlhl': [], 'hllh': [], 'hlll': [],
                          'lhhh': [], 'lhhl': [], 'lhlh': [], 'lhll': [],
                          'llhh': [], 'llhl': [], 'lllh': [], 'llll': []}

    # sort scores with index
    for group_name in GROUP_NAMES:
        eyes_cnn_scores_with_idx = []
        for i in range(len(property_scores['eyes_cnn_score'][group_name])):
            eyes_cnn_scores_with_idx.append([i, property_scores['eyes_cnn_score'][group_name][i]])

        mouth_cnn_scores_with_idx = []
        for i in range(len(property_scores['mouth_cnn_score'][group_name])):
            mouth_cnn_scores_with_idx.append([i, property_scores['mouth_cnn_score'][group_name][i]])

        pose_cnn_scores_with_idx = []
        for i in range(len(property_scores['pose_cnn_score'][group_name])):
            pose_cnn_scores_with_idx.append([i, property_scores['pose_cnn_score'][group_name][i]])

        eyes_cnn_scores_with_idx.sort(key=lambda x: x[1], reverse=True)
        mouth_cnn_scores_with_idx.sort(key=lambda x: x[1], reverse=True)
        pose_cnn_scores_with_idx.sort(key=lambda x: x[1], reverse=True)

        # generate largest/smallest score index info
        k = max(int(ratio * len(eyes_cnn_scores_with_idx)), 1)

        eyes_largest_idxs[group_name] = sorted([x[0] for x in eyes_cnn_scores_with_idx][:k])
        mouth_largest_idxs[group_name] = sorted([x[0] for x in mouth_cnn_scores_with_idx][:k])
        pose_largest_idxs[group_name] = sorted([x[0] for x in pose_cnn_scores_with_idx][:k])

        eyes_smallest_idxs[group_name] = sorted([x[0] for x in eyes_cnn_scores_with_idx][-k:])
        mouth_smallest_idxs[group_name] = sorted([x[0] for x in mouth_cnn_scores_with_idx][-k:])
        pose_smallest_idxs[group_name] = sorted([x[0] for x in pose_cnn_scores_with_idx][-k:])

    indices_info = {
        'eyes_largest': eyes_largest_idxs, 'eyes_smallest': eyes_smallest_idxs,
        'mouth_largest': mouth_largest_idxs, 'mouth_smallest': mouth_smallest_idxs,
        'pose_largest': pose_largest_idxs, 'pose_smallest': pose_smallest_idxs
    }

    return indices_info


# 각 핵심 속성 값 별 핵심 속성 값이 가장 큰 & 작은 k 장의 이미지에 대해 t-SNE 를 이용하여, 그룹별로 핵심 속성 값의 시각적 분포 파악
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - w_vectors_by_group (dict(NumPy array)) : sampling 된 intermediate w vector (각 그룹별)
# - indices_info       (dict)              : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 (그룹별) 인덱스 정보
#                                            {'eyes_largest': dict(list(int)), 'eyes_smallest': dict(list(int)),
#                                             'mouth_largest': dict(list(int)), 'mouth_smallest': dict(list(int)),
#                                             'pose_largest': dict(list(int)), 'pose_smallest': dict(list(int))}

# Returns:
# - stylegan/stylegan_vectorfind_v8/tsne_result 디렉토리에 그룹 별 & 각 핵심 속성 값 별 t-SNE 시각화 결과 저장

def run_tsne(w_vectors_by_group, indices_info):
    property_names = ['eyes', 'mouth', 'pose']

    tsne_result_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/tsne_result'
    os.makedirs(tsne_result_path, exist_ok=True)

    for property_name in property_names:
        for group_name in GROUP_NAMES:
            largest_img_idxs = indices_info[f'{property_name}_largest'][group_name]
            smallest_img_idxs = indices_info[f'{property_name}_smallest'][group_name]
            idxs = largest_img_idxs + smallest_img_idxs
            indexed_w_vectors = w_vectors_by_group[group_name][idxs]

            # run t-SNE
            print(f'running t-SNE for {property_name} / {group_name} ...')
            tsne_result = TSNE(n_components=2,
                               perplexity=min(50, len(indexed_w_vectors) - 1),
                               learning_rate=100,
                               n_iter=1000,
                               random_state=2025).fit_transform(indexed_w_vectors)

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
                             title=f't-SNE result of property {property_name} / {group_name}')

            fig.update_layout(width=720, height=600)
            fig.update_traces(marker=dict(size=2))
            fig.write_image(f'{tsne_result_path}/tsne_result_{property_name}_{group_name}.png')


# 핵심 속성 값의 변화를 나타내는 intermediate w vector 를 도출하기 위한 SVM 학습
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - latent_vectors_by_group (dict(NumPy array)) : sampling 된 intermediate w vector (각 그룹별)
# - group_name              (str)               : 머리 색, 머리 길이, 배경색 평균의 속성값 별 그룹명
#                                                 ('hhhh', 'hhhl', ..., 'llll')
# - indices_info            (dict)              : 각 핵심 속성 값이 가장 큰 & 가장 작은 k 장의 이미지의 (그룹별) 인덱스 정보
#                                                 {'eyes_largest': dict(list(int)), 'eyes_smallest': dict(list(int)),
#                                                  'mouth_largest': dict(list(int)), 'mouth_smallest': dict(list(int)),
#                                                  'pose_largest': dict(list(int)), 'pose_smallest': dict(list(int))}
# - svm_classifiers         (dict(dict(SVM)))   : 학습된 SVM (Support Vector Machine) 의 dict (각 그룹별)
#                                                 {'eyes': dict(SVM), 'mouth': dict(SVM), 'pose': dict(SVM)}

# Returns:
# - svm_classifiers              (dict)      : 학습된 SVM (Support Vector Machine) 의 dict (새로 학습된 SVM 을 추가하여 반환)
#                                              {'eyes': dict(list(SVM)),
#                                               'mouth': dict(list(SVM)),
#                                               'pose': dict(list(SVM))}
# - total_valid_cnt_info         (dict(int)) : 각 핵심 속성 값 별 전체 valid data 개수 정보
# - total_valid_correct_cnt_info (dict(int)) : 각 핵심 속성 값 별 SVM 예측 (largest / smallest) 이 맞은 valid data 개수 정보

def train_svm(latent_vectors_by_group, group_name, indices_info, svm_classifiers):
    train_ratio = 0.8
    total_valid_cnt_info = {}
    total_valid_correct_cnt_info = {}

    print(f'\ntraining SVM for {group_name} ...')
    for property_name in PROPERTY_NAMES:
        total_valid_cnt = 0
        total_valid_correct_cnt = 0
        svm_classifiers[property_name][group_name] = []

        for i in range(SVMS_PER_EACH_PROPERTY):

            # create dataset
            largest_img_idxs = indices_info[f'{property_name}_largest'][group_name]
            smallest_img_idxs = indices_info[f'{property_name}_smallest'][group_name]
            largest_img_idxs = random.sample(largest_img_idxs, len(largest_img_idxs))
            smallest_img_idxs = random.sample(smallest_img_idxs, len(smallest_img_idxs))

            largest_train_count = int(train_ratio * len(largest_img_idxs))
            smallest_train_count = int(train_ratio * len(smallest_img_idxs))

            train_idxs = largest_img_idxs[:largest_train_count] + smallest_img_idxs[:smallest_train_count]
            valid_idxs = largest_img_idxs[largest_train_count:] + smallest_img_idxs[smallest_train_count:]
            train_latent_vectors = latent_vectors_by_group[group_name][train_idxs]
            valid_latent_vectors = latent_vectors_by_group[group_name][valid_idxs]

            train_classes = ['largest'] * largest_train_count + ['smallest'] * smallest_train_count
            valid_classes = ['largest'] * (len(largest_img_idxs) - largest_train_count) + ['smallest'] * (len(smallest_img_idxs) - smallest_train_count)

            # train SVM
            svm_clf = svm.LinearSVC(random_state=2025+i)
            svm_classifier = svm_clf.fit(train_latent_vectors, train_classes)

            # valid SVM
            valid_predictions = svm_classifier.predict(valid_latent_vectors)

            # compute performance metric
            large_large = np.sum((np.array(valid_predictions) == 'largest') & (np.array(valid_classes) == 'largest'))
            large_small = np.sum((np.array(valid_predictions) == 'largest') & (np.array(valid_classes) == 'smallest'))
            small_large = np.sum((np.array(valid_predictions) == 'smallest') & (np.array(valid_classes) == 'largest'))
            small_small = np.sum((np.array(valid_predictions) == 'smallest') & (np.array(valid_classes) == 'smallest'))

            total_valid_cnt += large_large + large_small + small_large + small_small
            total_valid_correct_cnt += large_large + small_small

            svm_classifiers[property_name][group_name].append(svm_classifier)

        # compute accuracy of all SVMs for group
        accuracy = total_valid_correct_cnt / total_valid_cnt
        print(f'SVM accuracy of [{property_name}, {group_name}] (total valid count: {total_valid_cnt}): {accuracy:.4f}')

        total_valid_cnt_info[property_name] = total_valid_cnt
        total_valid_correct_cnt_info[property_name] = total_valid_correct_cnt

    return svm_classifiers, total_valid_cnt_info, total_valid_correct_cnt_info


# SVM 을 이용하여 핵심 속성 값의 변화를 나타내는 intermediate w vector 를 도출 (최종 w vector)
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - svm_classifiers (dict) : 학습된 SVM (Support Vector Machine) 의 dict (새로 학습된 SVM 을 추가하여 반환)
#                            {'eyes': dict(list(SVM)),
#                             'mouth': dict(list(SVM)),
#                             'pose': dict(list(SVM))}

# Returns:
# - property_score_vectors (dict) : 핵심 속성 값의 변화를 나타내는 intermediate w vector (각 그룹 별)
#                                   {'eyes_vector': dict(NumPy array),
#                                    'mouth_vector': dict(NumPy array),
#                                    'pose_vector': dict(NumPy array)}

def find_property_score_vectors(svm_classifiers):
    dim = ORIGINAL_HIDDEN_DIMS_W

    property_score_vectors = {}

    for property_name in PROPERTY_NAMES:
        property_score_vectors[f'{property_name}_vector'] = {}

        for group_name in GROUP_NAMES:
            classifiers = svm_classifiers[property_name][group_name]
            property_score_vectors[f'{property_name}_vector'][group_name] = []

            for classifier in classifiers:
                direction = classifier.coef_.reshape(1, dim).astype(np.float32)
                direction = direction / np.linalg.norm(direction)

                property_score_vectors[f'{property_name}_vector'][group_name].append(list(direction.flatten()))

            # convert to NumPy Array
            property_score_vectors[f'{property_name}_vector'][group_name] = (
                np.array(property_score_vectors[f'{property_name}_vector'][group_name]))

    return property_score_vectors


# 핵심 속성 값의 변화를 나타내는 intermediate w vector 에 대한 정보 저장
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - property_score_vectors (dict) : 핵심 속성 값의 변화를 나타내는 intermediate w vector (각 그룹 별)
#                                   {'eyes_vector': dict(NumPy array),
#                                    'mouth_vector': dict(NumPy array),
#                                    'pose_vector': dict(NumPy array)}

# Returns:
# - stylegan/stylegan_vectorfind_v8/property_score_vectors 디렉토리에 핵심 속성 값의 변화를 나타내는 intermediate w vector 정보 저장

def save_property_score_vectors_info(property_score_vectors):
    vector_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v8/property_score_vectors'
    os.makedirs(vector_save_dir, exist_ok=True)

    for group_name in GROUP_NAMES:
        eyes_vector_df = pd.DataFrame(property_score_vectors['eyes_vector'][group_name])
        mouth_vector_df = pd.DataFrame(property_score_vectors['mouth_vector'][group_name])
        pose_vector_df = pd.DataFrame(property_score_vectors['pose_vector'][group_name])

        eyes_vector_df.to_csv(f'{vector_save_dir}/eyes_change_w_vector_{group_name}.csv')
        mouth_vector_df.to_csv(f'{vector_save_dir}/mouth_change_w_vector_{group_name}.csv')
        pose_vector_df.to_csv(f'{vector_save_dir}/pose_change_w_vector_{group_name}.csv')


# StyleGAN-FineTune-v8 모델을 이용한 vector find 실시
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - finetune_v8_generator (nn.Module) : StyleGAN-FineTune-v8 의 Generator

def run_stylegan_vector_find(finetune_v8_generator, device):
    property_score_cnn = load_merged_property_score_cnn(device)

    # intermediate w vector 샘플링 & 핵심 속성 값이 가장 큰/작은 이미지 추출
    sampling_start_at = time.time()
    w_vectors_by_group, property_scores = sample_w_and_compute_property_scores(finetune_v8_generator,
                                                                               property_score_cnn)
    print(f'sampling (from latent vector w) running time (s) : {time.time() - sampling_start_at}\n')

    indices_info = extract_best_and_worst_k_images(property_scores)

    tsne_start_at = time.time()
    run_tsne(w_vectors_by_group, indices_info)
    print(f't-SNE running time (s) : {time.time() - tsne_start_at}')

    # SVM 학습 & 해당 SVM 으로 핵심 속성 값의 변화를 나타내는 최종 intermediate w vector 도출
    svm_classifiers = {'eyes': {}, 'mouth': {}, 'pose': {}}
    entire_valid_count = {'eyes': 0, 'mouth': 0, 'pose': 0}
    entire_valid_correct_count = {'eyes': 0, 'mouth': 0, 'pose': 0}

    svm_train_start_at = time.time()

    for group_name in GROUP_NAMES:
        svm_classifiers, total_valid_cnt_info, total_valid_correct_cnt_info =(
            train_svm(w_vectors_by_group, group_name, indices_info, svm_classifiers))

        for property_name in ['eyes', 'mouth', 'pose']:
            entire_valid_count[property_name] += total_valid_cnt_info[property_name]
            entire_valid_correct_count[property_name] += total_valid_correct_cnt_info[property_name]

    print('\n=== ENTIRE SVM ACCURACY ===')
    for property_name in PROPERTY_NAMES:
        entire_valid = entire_valid_count[property_name]
        entire_valid_correct = entire_valid_correct_count[property_name]
        entire_accuracy = entire_valid_correct / entire_valid
        print(f'entire accuracy for {property_name} : {entire_accuracy:.4f} ({entire_valid_correct} / {entire_valid})')
    print(f'\nSVM training running time (s) : {time.time() - svm_train_start_at}')

    property_score_vectors = find_property_score_vectors(svm_classifiers)
    save_property_score_vectors_info(property_score_vectors)
