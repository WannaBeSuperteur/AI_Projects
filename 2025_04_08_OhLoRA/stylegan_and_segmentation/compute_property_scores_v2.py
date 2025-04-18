import pandas as pd
import os
import cv2
import numpy as np
import math

np.set_printoptions(linewidth=160, suppress=True)

from compute_property_scores import read_parsing_result, normalize_all_scores, compute_background_mean_and_std


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PARSED_MAP_SIZE = 224
MIN_HAIR_PIXEL_THRESHOLD = 20


# 눈을 뜬 정도 (eyes) 및 고개 돌림 (pose) Score 계산
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - eyes_score (float) : 눈을 뜬 정도 Score
# - pose_score (float) : 고개 돌림 score

def compute_eyes_and_pose_score_v2(parsing_result):
    left_eye_area = parsing_result[PARSED_MAP_SIZE // 4 : 3 * PARSED_MAP_SIZE // 4, : 3 * PARSED_MAP_SIZE // 4]
    right_eye_area = parsing_result[PARSED_MAP_SIZE // 4 : 3 * PARSED_MAP_SIZE // 4, PARSED_MAP_SIZE // 4 :]

    left_eye_max_height_per_x = 0
    right_eye_max_height_per_x = 0

    left_eye_points = []
    right_eye_points = []

    # find max height of left and right eye
    for x in range(3 * PARSED_MAP_SIZE // 4):
        left_eye_min_y, left_eye_max_y = None, None
        right_eye_min_y, right_eye_max_y = None, None

        for y in range(PARSED_MAP_SIZE // 2):
            if left_eye_area[y][x] == 4:
                left_eye_points.append([y + PARSED_MAP_SIZE // 4, x])

                if left_eye_min_y is None:
                    left_eye_min_y = y
                    left_eye_max_y = y
                else:
                    left_eye_max_y = max(left_eye_max_y, y)

            if right_eye_area[y][x] == 5:
                right_eye_points.append([y + PARSED_MAP_SIZE // 4, x + PARSED_MAP_SIZE // 4])

                if right_eye_min_y is None:
                    right_eye_min_y = y
                    right_eye_max_y = y
                else:
                    right_eye_max_y = max(right_eye_max_y, y)

        left_eye_height = 0 if left_eye_min_y is None else left_eye_max_y - left_eye_min_y
        right_eye_height = 0 if right_eye_min_y is None else right_eye_max_y - right_eye_min_y

        left_eye_max_height_per_x = max(left_eye_max_height_per_x, left_eye_height)
        right_eye_max_height_per_x = max(right_eye_max_height_per_x, right_eye_height)

    # find point pair with max distance (for left eye)
    max_left_eye_distance = 0
    max_left_eye_distance_pair = None
    left_eye_angle_cos = 1.0

    for i in range(len(left_eye_points)):
        for j in range(i):
            left_eye_pt_0 = left_eye_points[i]
            left_eye_pt_1 = left_eye_points[j]
            squared_distance = (left_eye_pt_0[0] - left_eye_pt_1[0]) ** 2 + (left_eye_pt_0[1] - left_eye_pt_1[1]) ** 2

            if squared_distance > max_left_eye_distance:
                max_left_eye_distance = squared_distance
                max_left_eye_distance_pair = {'pt0': left_eye_pt_0, 'pt1': left_eye_pt_1}

    if max_left_eye_distance_pair is not None:
        dist_y = abs(max_left_eye_distance_pair['pt0'][0] - max_left_eye_distance_pair['pt1'][0])
        dist_x = abs(max_left_eye_distance_pair['pt0'][1] - max_left_eye_distance_pair['pt1'][1])
        dist_r = math.sqrt(dist_y ** 2 + dist_x ** 2)
        left_eye_angle_cos = dist_x / dist_r

    # find point pair with max distance (for right eye)
    max_right_eye_distance = 0
    max_right_eye_distance_pair = None
    right_eye_angle_cos = 1.0

    for i in range(len(right_eye_points)):
        for j in range(i):
            right_eye_pt_0 = right_eye_points[i]
            right_eye_pt_1 = right_eye_points[j]
            squared_distance = (right_eye_pt_0[0] - right_eye_pt_1[0])**2 + (right_eye_pt_0[1] - right_eye_pt_1[1])**2

            if squared_distance > max_right_eye_distance:
                max_right_eye_distance = squared_distance
                max_right_eye_distance_pair = {'pt0': right_eye_pt_0, 'pt1': right_eye_pt_1}

    if max_right_eye_distance_pair is not None:
        dist_y = abs(max_right_eye_distance_pair['pt0'][0] - max_right_eye_distance_pair['pt1'][0])
        dist_x = abs(max_right_eye_distance_pair['pt0'][1] - max_right_eye_distance_pair['pt1'][1])
        dist_r = math.sqrt(dist_y ** 2 + dist_x ** 2)
        right_eye_angle_cos = dist_x / dist_r

    # compute eyes score
    left_eye_score = left_eye_max_height_per_x * left_eye_angle_cos
    right_eye_score = right_eye_max_height_per_x * right_eye_angle_cos
    eyes_score = max(left_eye_score, right_eye_score)

    # compute pose score
    if len(left_eye_points) == 0 or len(right_eye_points) == 0:
        pose_score = 0.0

    else:
        left_eye_points_mean = np.mean(np.array(left_eye_points), axis=0)
        right_eye_points_mean = np.mean(np.array(right_eye_points), axis=0)

        eyes_dist_y = right_eye_points_mean[0] - left_eye_points_mean[0]
        eyes_dist_x = right_eye_points_mean[1] - left_eye_points_mean[1]
        pose_score = eyes_dist_y / eyes_dist_x

    return eyes_score, pose_score


# 머리 색 (hair_color) 의 밝기 Score 계산
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - parsing_result       (np.array) : Parsing Result (224 x 224)
# - face_detection_image (np.array) : Segmentation 결과에서 Face Detection 처리된 원본 이미지 (224 x 224)

# Returns:
# - hair_color_score (float) : 머리 색 Score

def compute_hair_color_score_v2(parsing_result, face_detection_image):
    hair_color_info = []

    for y in range(PARSED_MAP_SIZE):
        for x in range(PARSED_MAP_SIZE):
            if parsing_result[y][x] == 10:
                hair_color_info.append(face_detection_image[y][x])

    hair_color_info = np.array(hair_color_info)
    hair_color_pixel_count = len(hair_color_info)

    if len(hair_color_info) == 0:
        return 127.5

    hair_color_rgb_mean = np.mean(hair_color_info, axis=1)
    hair_color_rgb_mean = np.sort(hair_color_rgb_mean)
    hair_color_rgb_mean = hair_color_rgb_mean[int(0.1 * hair_color_pixel_count):int(0.9 * hair_color_pixel_count)]

    if len(hair_color_rgb_mean) == 0:
        return 127.5

    hair_color_score = np.mean(hair_color_rgb_mean)

    return hair_color_score


# 머리 길이 (hair_length) Score 계산
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - hair_length_score (float) : 머리 길이 Score

def compute_hair_length_score_v2(parsing_result):

    hair_y_count = 0
    additional_estimated_hair_len = 0

    for y in range(PARSED_MAP_SIZE // 4, PARSED_MAP_SIZE):
        if np.count_nonzero(parsing_result[y] == 10) >= MIN_HAIR_PIXEL_THRESHOLD:
            hair_y_count += 1

    if np.count_nonzero(parsing_result[PARSED_MAP_SIZE - 1] == 10) >= MIN_HAIR_PIXEL_THRESHOLD:
        left_half = parsing_result[PARSED_MAP_SIZE - 1][: PARSED_MAP_SIZE // 2]
        right_half = parsing_result[PARSED_MAP_SIZE - 1][PARSED_MAP_SIZE // 2 :]

        left_hair_point_count = np.count_nonzero(left_half == 10)
        right_hair_point_count = np.count_nonzero(right_half == 10)

        additional_estimated_left_hair_len = max(left_hair_point_count - MIN_HAIR_PIXEL_THRESHOLD // 2, 0) / 2.0
        additional_estimated_right_hair_len = max(right_hair_point_count - MIN_HAIR_PIXEL_THRESHOLD // 2, 0) / 2.0
        additional_estimated_hair_len = additional_estimated_left_hair_len + additional_estimated_right_hair_len

    return hair_y_count + additional_estimated_hair_len


# 입을 벌린 정도 (mouth) Score 계산
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - mouth_score (float) : 입을 벌린 정도 Score

def compute_mouth_score_v2(parsing_result):

    lips_pixels = 0
    mouth_pixels = 0

    for y in range(PARSED_MAP_SIZE):
        upper_lips_pixels_y = np.count_nonzero(parsing_result[y] == 7)
        lower_lips_pixels_y = np.count_nonzero(parsing_result[y] == 9)
        mouth_pixels_y = np.count_nonzero(parsing_result[y] == 8)

        lips_pixels += (upper_lips_pixels_y + lower_lips_pixels_y)
        mouth_pixels += mouth_pixels_y

    if lips_pixels + mouth_pixels == 0:
        return 0
    else:
        mouth_score = mouth_pixels / (lips_pixels + mouth_pixels)

    return mouth_score


# 생성된 이미지 중 필터링된 모든 이미지를 읽어서 그 이미지의 모든 Score 를 산출
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - all_img_nos (list(int)) : Parsing Result (224 x 224) 에 대응되는 원본 이미지들의 번호 (No.) 의 리스트

# Returns:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과
#                                   columns = ['img_no', 'img_path', 'eyes_score', ..., 'background_std_score']
#                                   img_path 는 stylegan_and_segmentation/stylegan/synthesize_results_filtered 기준

def compute_all_image_scores(all_img_nos):
    generated_img_dir_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
    img_paths = [f'{generated_img_dir_path}/{img_no:06d}.jpg' for img_no in all_img_nos]

    parsing_result_dir_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/segmentation_results'
    parsing_result_paths = [f'{parsing_result_dir_path}/parsing_{img_no:06d}.png' for img_no in all_img_nos]
    face_detection_result_paths = [f'{parsing_result_dir_path}/face_{img_no:06d}.png' for img_no in all_img_nos]

    all_scores_dict = {'img_no': all_img_nos,
                       'img_path': img_paths,
                       'eyes_score': [],
                       'hair_color_score': [],
                       'hair_length_score': [],
                       'mouth_score': [],
                       'pose_score': [],
                       'background_mean_score': [],
                       'background_std_score': []}

    for idx, (img_path, parsing_result_path, face_detection_result_path) \
            in enumerate(zip(img_paths, parsing_result_paths, face_detection_result_paths)):

        if idx < 10 or idx % 100 == 0:
            print(f'scoring image {idx + 1} / {len(parsing_result_paths)} ...')

        # read parsing result
        parsing_result = read_parsing_result(parsing_result_path)
        face_detection_image = cv2.imdecode(np.fromfile(face_detection_result_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # compute property scores
        eyes_score, pose_score = compute_eyes_and_pose_score_v2(parsing_result)
        hair_color_score = compute_hair_color_score_v2(parsing_result, face_detection_image)
        hair_length_score = compute_hair_length_score_v2(parsing_result)
        mouth_score = compute_mouth_score_v2(parsing_result)
        background_mean, background_std = compute_background_mean_and_std(parsing_result, face_detection_image)

        # append to all_scores result
        all_scores_dict['eyes_score'].append(eyes_score)
        all_scores_dict['hair_color_score'].append(hair_color_score)
        all_scores_dict['hair_length_score'].append(hair_length_score)
        all_scores_dict['mouth_score'].append(mouth_score)
        all_scores_dict['pose_score'].append(pose_score)
        all_scores_dict['background_mean_score'].append(background_mean)
        all_scores_dict['background_std_score'].append(background_std)

    all_scores = pd.DataFrame(all_scores_dict)
    return all_scores


if __name__ == '__main__':
    img_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
    img_names = os.listdir(img_dir)
    img_nos = [int(img_name[:-4]) for img_name in img_names]

    all_scores = compute_all_image_scores(img_nos)
    all_scores = normalize_all_scores(all_scores)

    all_scores_dir_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/property_score_results'
    os.makedirs(all_scores_dir_path, exist_ok=True)

    all_scores_path = f'{all_scores_dir_path}/all_scores_v2.csv'
    all_scores.to_csv(all_scores_path)
