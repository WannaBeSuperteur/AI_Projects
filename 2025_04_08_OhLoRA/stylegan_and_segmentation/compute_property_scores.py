import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Segmentation 결과에서 Parsing Result 읽기
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result_path (str) : Parsing Result 경로

# Returns:
# - parsing_result (np.array) : Parsing Result (224 x 224)

def read_parsing_result(parsing_result_path):
    raise NotImplementedError


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


# 배경색의 밝기 (background_light) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - background_light (float) : 배경색의 밝기 Score

def compute_background_light_score(parsing_result):
    raise NotImplementedError


# 배경색의 표준편차 (background_std) Score 계산
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - parsing_result (np.array) : Parsing Result (224 x 224)

# Returns:
# - background_std (float) : 배경색의 밝기 Score

def compute_background_std_score(parsing_result):
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


# 이미지를 읽어서 그 이미지의 모든 Score 를 산출
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - all_img_nos (list(int)) : Parsing Result (224 x 224) 에 대응되는 원본 이미지들의 번호 (No.) 의 리스트

# Returns:
# - all_scores (Pandas DataFrame) : 모든 이미지에 대한 모든 Score 의 계산 결과
#                                   columns = ['img_no', 'img_path', 'eyes_score', ..., 'pose_score']
#                                   img_path 는 stylegan_and_segmentation/stylegan/synthesize_results_filtered 기준

def compute_all_image_scores(all_img_nos):
    raise NotImplementedError


if __name__ == '__main__':
    img_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
    img_names = os.listdir(img_dir)
    img_nos = [int(img_name[:-4]) for img_name in img_names]

    all_scores = compute_all_image_scores(img_nos)
    all_scores_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/property_score_results/all_scores.csv'
    all_scores.to_csv(all_scores_path)
