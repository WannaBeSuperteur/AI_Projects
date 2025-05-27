import pandas as pd
import os
import cv2
import numpy as np
np.set_printoptions(linewidth=160, suppress=True)

import random
random.seed(2025)

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PARSED_MAP_SIZE = 224


# Segmentation 결과에서 Parsing Result 읽기
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - parsing_result_path (str) : Parsing Result 경로

# Returns:
# - parsing_result (np.array) : Parsing Result (224 x 224)

def read_parsing_result(parsing_result_path):
    parsing_result = cv2.imdecode(np.fromfile(parsing_result_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return parsing_result


# 모든 이미지의 모든 핵심 속성 값 Score 를 정규화
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과
#                                   columns = ['img_no', 'img_path', 'hairstyle_score']

# Returns:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과 (정규화된)

def normalize_all_scores(all_scores):
    scores = pd.DataFrame(all_scores)
    properties = ['hairstyle_score']

    for p in properties:
        property_mean = scores[p].mean()
        property_std = scores[p].std()
        scores[p] = (scores[p] - property_mean) / property_std

        scores[p] = scores[p].apply(lambda x: round(x, 4))

    return scores


# 곱슬머리 vs. 직모 (hair_style) Score 계산 (idea 1 -> 미 사용, 정확도 낮음)
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - parsing_result       (np.array) : Parsing Result (224 x 224)
# - face_detection_image (np.array) : Segmentation 결과에서 Face Detection 처리된 원본 이미지 (224 x 224)

# Returns:
# - hairstyle_score (float) : 곱슬머리 vs. 직모 (hair_style) Score

def compute_hairstyle_score_idea_1(parsing_result, face_detection_image):
    slope_dy = 20

    left_area = parsing_result[PARSED_MAP_SIZE // 4:, :PARSED_MAP_SIZE // 2]
    right_area = parsing_result[PARSED_MAP_SIZE // 4:, PARSED_MAP_SIZE // 2:]

    left_area_hair_ends = []
    right_area_hair_starts = []
    hair_pixel_rgb_means = []

    # compute hair pixel RGB mean (= brightness) std
    for y in range(PARSED_MAP_SIZE // 4, PARSED_MAP_SIZE):
        for x in range(PARSED_MAP_SIZE):
            if parsing_result[y][x] == 10:
                hair_pixel_rgb_means.append(np.mean(face_detection_image[y][x]))

    hair_pixel_rgb_std = np.std(hair_pixel_rgb_means)

    # check hair end x-pos values (left) / hair start x-pose values (right) to compute 'slope' of hair area
    for y in range(3 * PARSED_MAP_SIZE // 4):

        # for left area
        left_area_hair_end = None
        for x in range(1, PARSED_MAP_SIZE // 2 - 2):
            if left_area[y][x - 1] == 10 and left_area[y][x] == 10 and left_area[y][x + 1] != 10 and left_area[y][x + 2] != 10:
                left_area_hair_end = x
                break

        left_area_hair_ends.append(left_area_hair_end)

        # for right area
        right_area_hair_start = None
        for x in range(1, PARSED_MAP_SIZE // 2 - 2):
            if right_area[y][x - 1] != 10 and right_area[y][x] != 10 and right_area[y][x + 1] == 10 and right_area[y][x + 2] == 10:
                right_area_hair_start = x

        right_area_hair_starts.append(right_area_hair_start)

    # compute 'slope' of hair area
    left_area_slope = []
    right_area_slope = []

    for y in range(3 * PARSED_MAP_SIZE // 4 - (slope_dy + 1)):

        # for left area
        hair_end_0 = left_area_hair_ends[y]
        hair_end_1 = left_area_hair_ends[y + slope_dy]

        if hair_end_0 is None or hair_end_1 is None:
            left_area_slope.append(None)
        else:
            left_area_slope.append((hair_end_1 - hair_end_0) / slope_dy)

        # for right area
        hair_start_0 = right_area_hair_starts[y]
        hair_start_1 = right_area_hair_starts[y + slope_dy]

        if hair_start_0 is None or hair_start_1 is None:
            right_area_slope.append(None)
        else:
            right_area_slope.append((hair_start_1 - hair_start_0) / slope_dy)

    # compute hairstyle score
    left_area_hairstyle_scores = []
    right_area_hairstyle_scores = []

    for y in range(3 * PARSED_MAP_SIZE // 4 - (slope_dy + 1)):
        y_parse_result = y + PARSED_MAP_SIZE // 4

        # for left area
        if left_area_slope[y] is not None:
            for x in range(PARSED_MAP_SIZE // 2):
                dx = round(slope_dy * left_area_slope[y])

                if x + dx < 0 or x + dx >= PARSED_MAP_SIZE:
                    continue

                if parsing_result[y_parse_result][x] == 10 and parsing_result[y_parse_result + slope_dy][x + dx] == 10:
                    rgb_mean_0 = np.mean(face_detection_image[y_parse_result][x])
                    rgb_mean_1 = np.mean(face_detection_image[y_parse_result + slope_dy][x + dx])
                    left_area_hairstyle_scores.append(abs(rgb_mean_0 - rgb_mean_1))

        # for right area
        if right_area_slope[y] is not None:
            for x in range(PARSED_MAP_SIZE // 2, PARSED_MAP_SIZE):
                dx = round(slope_dy * right_area_slope[y])

                if x + dx < 0 or x + dx >= PARSED_MAP_SIZE:
                    continue

                if parsing_result[y_parse_result][x] == 10 and parsing_result[y_parse_result + slope_dy][x + dx] == 10:
                    rgb_mean_0 = np.mean(face_detection_image[y_parse_result][x])
                    rgb_mean_1 = np.mean(face_detection_image[y_parse_result + slope_dy][x + dx])
                    right_area_hairstyle_scores.append(abs(rgb_mean_0 - rgb_mean_1))

    # compute final score
    hairstyle_scores = left_area_hairstyle_scores + right_area_hairstyle_scores
    hairstyle_score = np.mean(hairstyle_scores) / hair_pixel_rgb_std

    return hairstyle_score


# 곱슬머리 vs. 직모 (hair_style) Score 계산 (idea 1 -> 미 사용)
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - parsing_result       (np.array) : Parsing Result (224 x 224)
# - face_detection_image (np.array) : Segmentation 결과에서 Face Detection 처리된 원본 이미지 (224 x 224)

# Returns:
# - hairstyle_score (float) : 곱슬머리 vs. 직모 (hair_style) Score

def compute_hairstyle_score(parsing_result, face_detection_image):
    dx, dy = 8, 8
    sx, sy = 4, 4

    # compute hairstyle score
    rgb_mean_diff_ratios = []

    for x in range(dx, PARSED_MAP_SIZE - (dx + sx - 1)):
        for y in range(PARSED_MAP_SIZE // 2, PARSED_MAP_SIZE - (dy + sy - 1)):

            is_hair_check_0 = (parsing_result[y][x] == 10 and
                               parsing_result[y + dy][x] == 10 and
                               parsing_result[y][x - dx] == 10 and
                               parsing_result[y][x + dx] == 10)

            is_hair_check_1 = (parsing_result[y + (sy - 1)][x + (sx - 1)] == 10 and
                               parsing_result[y + dy + (sy - 1)][x + (sx - 1)] == 10 and
                               parsing_result[y + (sy - 1)][x - dx - (sx - 1)] == 10 and
                               parsing_result[y + (sy - 1)][x + dx + (sx - 1)] == 10)

            if is_hair_check_0 and is_hair_check_1:
                rgb_mean = np.mean(face_detection_image[y:y + sy, x:x + sx])
                rgb_mean_bottom = np.mean(face_detection_image[y + dy:y + dy + sy, x:x + sx])
                rgb_mean_left = np.mean(face_detection_image[y:y + sy, x - dx:x - dx + sx])
                rgb_mean_right = np.mean(face_detection_image[y:y + sy, x + dx:x + dx + sx])

                diff_horizontal = max(abs(rgb_mean - rgb_mean_left), abs(rgb_mean - rgb_mean_right))
                diff_vertical = abs(rgb_mean - rgb_mean_bottom)
                diff_ratio = diff_vertical / (diff_horizontal + 5.0)
                rgb_mean_diff_ratios.append(diff_ratio)

    if len(rgb_mean_diff_ratios) >= 1:
        hairstyle_score = np.mean(rgb_mean_diff_ratios)
    else:
        hairstyle_score = 0.45

    return hairstyle_score


# 생성된 이미지 중 필터링된 모든 이미지를 읽어서 그 이미지의 모든 Score 를 산출
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - all_img_nos (list(int)) : Parsing Result (224 x 224) 에 대응되는 원본 이미지들의 번호 (No.) 의 리스트

# Returns:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과
#                                   columns = ['img_no', 'img_path', 'hairstyle_score']

def compute_all_image_scores(all_img_nos):
    generated_img_dir_path = f'{PROJECT_DIR_PATH}/stylegan/generated_face_images_filtered'
    img_paths = [f'{generated_img_dir_path}/{img_no:06d}.jpg' for img_no in all_img_nos]

    parsing_result_dir_path = f'{PROJECT_DIR_PATH}/property_score_cnn/segmentation/segmentation_results'
    parsing_result_paths = [f'{parsing_result_dir_path}/parsing_{img_no:06d}.png' for img_no in all_img_nos]
    face_detection_result_paths = [f'{parsing_result_dir_path}/face_{img_no:06d}.png' for img_no in all_img_nos]

    all_scores_dict = {'img_no': all_img_nos,
                       'img_path': img_paths,
                       'hairstyle_score': []}

    for idx, (img_path, parsing_result_path, face_detection_result_path) \
            in enumerate(zip(img_paths, parsing_result_paths, face_detection_result_paths)):

        if idx < 10 or idx % 100 == 0:
            print(f'scoring image {idx + 1} / {len(parsing_result_paths)} ...')

        # read parsing result
        parsing_result = read_parsing_result(parsing_result_path)
        face_detection_image = cv2.imdecode(np.fromfile(face_detection_result_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # compute property score (hairstyle score only)
        hairstyle_score = compute_hairstyle_score(parsing_result, face_detection_image)

        # append to all_scores result
        all_scores_dict['hairstyle_score'].append(hairstyle_score)

    all_scores = pd.DataFrame(all_scores_dict)
    return all_scores


if __name__ == '__main__':
    img_dir = f'{PROJECT_DIR_PATH}/stylegan/generated_face_images_filtered'
    img_names = os.listdir(img_dir)
    img_nos = [int(img_name[:-4]) for img_name in img_names]

    all_scores = compute_all_image_scores(img_nos)
    all_scores = normalize_all_scores(all_scores)

    all_scores_dir_path = f'{PROJECT_DIR_PATH}/property_score_cnn/segmentation/property_score_results'
    os.makedirs(all_scores_dir_path, exist_ok=True)

    all_scores_path = f'{all_scores_dir_path}/all_scores_ohlora_v3.csv'
    all_scores.to_csv(all_scores_path)
