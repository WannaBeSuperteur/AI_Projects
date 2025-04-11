import pandas as pd
import os
import cv2
import numpy as np
import time

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PARSED_MAP_SIZE = 224


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
    left_eye_area = parsing_result[PARSED_MAP_SIZE // 4 : 3 * PARSED_MAP_SIZE // 4, : 3 * PARSED_MAP_SIZE // 4]
    right_eye_area = parsing_result[PARSED_MAP_SIZE // 4 : 3 * PARSED_MAP_SIZE // 4, PARSED_MAP_SIZE // 4 :]

    left_eye_min_y, left_eye_max_y = None, None
    right_eye_min_y, right_eye_max_y = None, None

    for y in range(3 * PARSED_MAP_SIZE // 8):
        if 4 in left_eye_area[y]:
            if left_eye_min_y is None:
                left_eye_min_y = y
                left_eye_max_y = y
            else:
                left_eye_max_y = max(left_eye_max_y, y)

        if 5 in right_eye_area[y]:
            if right_eye_min_y is None:
                right_eye_min_y = y
                right_eye_max_y = y
            else:
                right_eye_max_y = max(right_eye_max_y, y)

    left_eye_height = 0 if left_eye_min_y is None else left_eye_max_y - left_eye_min_y
    right_eye_height = 0 if right_eye_min_y is None else right_eye_max_y - right_eye_min_y

    return (left_eye_height + right_eye_height) / 2.0


# 머리 색 (hair_color) 의 밝기 Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)
# - image          (np.array) : 원본 이미지 (224 x 224)

# Returns:
# - hair_color_score (float) : 머리 색 Score

def compute_hair_color_score(parsing_result, image):
    hair_color_info = []

    for y in range(PARSED_MAP_SIZE):
        for x in range(PARSED_MAP_SIZE):
            if parsing_result[y][x] == 10:
                hair_color_info.append(image[y][x])

    hair_color_info = np.array(hair_color_info)

    hair_color_rgb_mean = np.mean(hair_color_info, axis=1)
    hair_color_score = np.median(hair_color_rgb_mean)
    return hair_color_score


# 머리 길이 (hair_length) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - hair_length_score (float) : 머리 길이 Score

def compute_hair_length_score(parsing_result):
    hair_min_y, hair_max_y = None, None

    for y in range(PARSED_MAP_SIZE):
        if 10 in parsing_result[y]:
            if hair_min_y is None:
                hair_min_y = y
                hair_max_y = y
            else:
                hair_max_y = max(hair_max_y, y)

    if hair_max_y == PARSED_MAP_SIZE - 1:
        left_half = parsing_result[PARSED_MAP_SIZE - 1][: PARSED_MAP_SIZE // 2]
        right_half = parsing_result[PARSED_MAP_SIZE - 1][PARSED_MAP_SIZE // 2 :]

        left_hair_point_count = np.count_nonzero(left_half == 10)
        right_hair_point_count = np.count_nonzero(right_half == 10)

        additional_estimated_hair_length = (left_hair_point_count + right_hair_point_count) / 2.0
        return hair_max_y + additional_estimated_hair_length

    else:
        return hair_max_y


# 입을 벌린 정도 (mouth) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - mouth_score (float) : 입을 벌린 정도 Score

def compute_mouth_score(parsing_result):
    lips_min_y, lips_max_y = None, None
    mouth_min_y, mouth_max_y = None, None

    for y in range(PARSED_MAP_SIZE):
        if 7 in parsing_result[y] or 9 in parsing_result[y]:
            if lips_min_y is None:
                lips_min_y = y
                lips_max_y = y
            else:
                lips_max_y = max(lips_max_y, y)

        if 8 in parsing_result[y]:
            if mouth_min_y is None:
                mouth_min_y = y
                mouth_max_y = y
            else:
                mouth_max_y = max(mouth_max_y, y)

    lips_height = 0 if lips_min_y is None else lips_max_y - lips_min_y
    mouth_height = 0 if mouth_min_y is None else mouth_max_y - mouth_min_y

    mouth_score = lips_height + mouth_height
    return mouth_score


# 고개 돌림 (pose) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - pose_score (float) : 고개 돌림 Score (왼쪽: -1 ~ 정면: 0 ~ 오른쪽: +1)

def compute_pose_score(parsing_result):
    nose_xs = []
    nose_ys = []

    for y in range(PARSED_MAP_SIZE):
        for x in range(PARSED_MAP_SIZE):
            if parsing_result[y][x] == 6:
                nose_xs.append(x)
                nose_ys.append(y)

    corr = np.corrcoef(nose_xs, nose_ys)[0][1]
    pose_score = (-1.0) * corr
    return pose_score


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

    for idx, (img_path, parsing_result_path) in enumerate(zip(img_paths, parsing_result_paths)):
        if idx < 10 or idx % 100 == 0:
            print(f'scoring image {idx + 1} / {len(parsing_result_paths)} ...')

        # read parsing result
        parsing_result = read_parsing_result(parsing_result_path)
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(PARSED_MAP_SIZE, PARSED_MAP_SIZE), interpolation=cv2.INTER_LINEAR)

        # compute property scores
        eyes_score = compute_eyes_score(parsing_result)
        hair_color_score = compute_hair_color_score(parsing_result, image)
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

    property_to_apply_minmax = ['eyes_score', 'hair_color_score', 'hair_length_score', 'mouth_score']  # 0 ~ 1
    property_to_apply_m1_to_p1 = ['pose_score']  # -1 ~ +1

    for p in property_to_apply_minmax:
        scores[p] = (scores[p] - scores[p].min()) / (scores[p].max() - scores[p].min())
        scores[p] = scores[p].apply(lambda x: round(x, 4))

    for p in property_to_apply_m1_to_p1:
        scores[p] = 2.0 * (scores[p] - scores[p].min()) / (scores[p].max() - scores[p].min()) - 1.0
        scores[p] = scores[p].apply(lambda x: round(x, 4))

    return scores


if __name__ == '__main__':
    img_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
    img_names = os.listdir(img_dir)
    img_nos = [int(img_name[:-4]) for img_name in img_names]

    all_scores = compute_all_image_scores(img_nos)
    all_scores = normalize_all_scores(all_scores)

    all_scores_dir_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/property_score_results'
    os.makedirs(all_scores_dir_path, exist_ok=True)

    all_scores_path = f'{all_scores_dir_path}/all_scores.csv'
    all_scores.to_csv(all_scores_path)
