import pandas as pd
import os
import cv2
import numpy as np

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Segmentation 결과에서 Parsing Result 읽기
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result_path (str) : Parsing Result 경로

# Returns:
# - parsing_result (np.array) : Parsing Result (224 x 224)

def read_parsing_result(parsing_result_path):
    parsing_result = cv2.imdecode(np.fromfile(parsing_result_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return parsing_result


# 눈을 뜬 정도 (eyes) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - eyes_score (float) : 눈을 뜬 정도 Score

def compute_eyes_score(parsing_result):
    raise NotImplementedError


# 머리 색 (hair_color) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - hair_color_score (float) : 머리 색 Score

def compute_hair_color_score(parsing_result):
    raise NotImplementedError


# 머리 길이 (hair_length) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - hair_length_score (float) : 머리 길이 Score

def compute_hair_length_score(parsing_result):
    raise NotImplementedError


# 입을 벌린 정도 (mouth) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - mouth_score (float) : 입을 벌린 정도 Score

def compute_mouth_score(parsing_result):
    raise NotImplementedError


# 고개 돌림 (pose) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - pose_score (float) : 고개 돌림 Score

def compute_pose_score(parsing_result):
    raise NotImplementedError


# 생성된 이미지 중 필터링된 모든 이미지를 읽어서 그 이미지의 모든 Score 를 산출
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - all_img_nos (list(int)) : Parsing Result (224 x 224) 에 대응되는 원본 이미지들의 번호 (No.) 의 리스트

# Returns:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과
#                                   columns = ['img_no', 'img_path', 'eyes_score', ..., 'pose_score']
#                                   img_path 는 stylegan_and_segmentation/stylegan/synthesize_results_filtered 기준

def compute_all_image_scores(all_img_nos):
    generated_img_dir_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
    img_paths = [f'{generated_img_dir_path}/{img_no:06d}.jpg' for img_no in all_img_nos]

    parsing_result_dir_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/segmentation_results'
    parsing_result_paths = [f'{parsing_result_dir_path}/parsing_{img_no:06d}.png' for img_no in all_img_nos]

    all_scores_dict = {'img_no': all_img_nos,
                       'img_path': img_paths,
                       'eyes_score': [],
                       'hair_color_score': [],
                       'hair_length_score': [],
                       'mouth_score': [],
                       'pose_score': []}

    for idx, parsing_result_path in enumerate(parsing_result_paths):
        if idx < 10 or idx % 100 == 0:
            print(f'scoring image {idx + 1} / {len(parsing_result_paths)} ...')

        # read parsing result
        parsing_result = read_parsing_result(parsing_result_path)

        # compute property scores
        eyes_score = compute_eyes_score(parsing_result)
        hair_color_score = compute_hair_color_score(parsing_result)
        hair_length_score = compute_hair_length_score(parsing_result)
        mouth_score = compute_mouth_score(parsing_result)
        pose_score = compute_pose_score(parsing_result)

        # append to all_scores result
        all_scores_dict['eyes_score'].append(eyes_score)
        all_scores_dict['hair_color_score'].append(hair_color_score)
        all_scores_dict['hair_length_score'].append(hair_length_score)
        all_scores_dict['mouth_score'].append(mouth_score)
        all_scores_dict['pose_score'].append(pose_score)

    all_scores = pd.DataFrame(all_scores_dict)
    return all_scores


# 모든 이미지의 모든 핵심 속성 값 Score 를 정규화
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과
#                                   columns = ['img_no', 'img_path', 'eyes_score', ..., 'pose_score']
#                                   img_path 는 stylegan_and_segmentation/stylegan/synthesize_results_filtered 기준

# Returns:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과 (정규화된)

def normalize_all_scores(all_scores):
    scores = pd.DataFrame(all_scores)

    property_to_apply_minmax = ['eyes_score', 'hair_color_score', 'hair_length_score', 'mouth_score']
    property_to_apply_m1_to_p1 = ['pose_score']

    for p in property_to_apply_minmax:
        scores[p] = (scores[p] - scores[p].min()) / (scores[p].max() - scores[p].min())

    for p in property_to_apply_m1_to_p1:
        scores[p] = 2.0 * (scores[p] - scores[p].min()) / (scores[p].max() - scores[p].min()) - 1.0

    return scores


if __name__ == '__main__':
    img_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
    img_names = os.listdir(img_dir)
    img_nos = [int(img_name[:-4]) for img_name in img_names]

    all_scores = compute_all_image_scores(img_nos)
    all_scores = normalize_all_scores(all_scores)

    all_scores_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/property_score_results/all_scores.csv'
    all_scores.to_csv(all_scores_path)
