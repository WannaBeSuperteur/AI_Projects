
import os
import cv2
import numpy as np
import pandas as pd

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MAX_DATA_SIZE_PER_CLASS = 200


# 특정 이미지의 조건 충족 여부 확인 후, 조건 충족 시 Sub Dataset 안에 저장
# Create Date : 2026.03.19
# Last Update Date : -

# Arguments:
# - img                  (np.array)             : OpenCV 로 읽은 학습/테스트 데이터 이미지
# - avg_pixel_hsv_ranges (list([float, float])) : 특정 이미지의 모든 픽셀을 channel-wise 하게 평균한 픽셀의 색상,채도,명도 값 범위
#                                                 (색상, 채도, 명도 각각 sub-list)
# - std_ranges           (list([float, float])) : 특정 이미지의 모든 픽셀에 대한 R, G, B 값 각각의 표준편차 범위
#                                                 (R, G, B 각각 sub-list)
# - save_path            (str)                  : 이미지 저장 경로 (sub-dataset)

# Returns
# - is_saved (bool) : 이미지 저장 여부 (= 조건 충족 여부)

def check_condition_and_save_image_in_subdataset(img, avg_pixel_hsv_ranges, std_ranges, save_path):
    channel_mean = np.mean(img, axis=(0, 1))
    channel_std = np.std(img, axis=(0, 1))

    # RGB image
    if img.shape[2] == 3:
        hsv = cv2.cvtColor(np.array([[channel_mean]], dtype=np.uint8), cv2.COLOR_BGR2HSV)

        hue_in_range = avg_pixel_hsv_ranges[0][0] <= hsv[0][0][0] <= avg_pixel_hsv_ranges[0][1]
        saturation_in_range = avg_pixel_hsv_ranges[1][0] <= hsv[0][0][1] <= avg_pixel_hsv_ranges[1][1]
        value_in_range = avg_pixel_hsv_ranges[2][0] <= hsv[0][0][2] <= avg_pixel_hsv_ranges[2][1]
        hsv_in_range = hue_in_range and saturation_in_range and value_in_range

        std_in_range_r = std_ranges[0][0] <= channel_std[2] <= std_ranges[0][1]
        std_in_range_g = std_ranges[1][0] <= channel_std[1] <= std_ranges[1][1]
        std_in_range_b = std_ranges[2][0] <= channel_std[0] <= std_ranges[2][1]
        std_in_range = std_in_range_r and std_in_range_g and std_in_range_b

    # Grayscale image
    else:
        hsv_in_range = avg_pixel_hsv_ranges[0][0] <= channel_mean[0] <= avg_pixel_hsv_ranges[0][1]
        std_in_range = std_ranges[0][0] <= channel_std[0] <= std_ranges[0][1]

    if hsv_in_range and std_in_range:
        cv2.imwrite(save_path, img)
        return True
    return False


# 특정 이미지의 모든 픽셀을 channel-wise 하게 평균한 픽셀의 색상,채도,명도 값 + 모든 픽셀에 대한 R, G, B 값 각각의 표준편차 계산
# Create Date : 2026.03.19
# Last Update Date : -

# Arguments:
# - img (np.array) : OpenCV 로 읽은 학습/테스트 데이터 이미지

# Returns
# - compute_image_condition_values (dict(float)) : 특정 이미지의 모든 픽셀을 channel-wise 하게 평균한 픽셀의 색상,채도,명도 값 +
#                                                  특정 이미지의 모든 픽셀에 대한 R, G, B 값 각각의 표준편차 값

def compute_image_condition_values(img):
    channel_mean = np.mean(img, axis=(0, 1))
    channel_std = np.std(img, axis=(0, 1))

    # RGB image
    if img.shape[2] == 3:
        hsv = cv2.cvtColor(np.array([[channel_mean]], dtype=np.uint8), cv2.COLOR_BGR2HSV)
        hue = hsv[0][0][0]
        saturation = hsv[0][0][1]
        value_ = hsv[0][0][2]
        std_r = round(channel_std[2], 2)
        std_g = round(channel_std[1], 2)
        std_b = round(channel_std[0], 2)

        return {'hue': hue, 'saturation': saturation, 'value': value_, 'std_r': std_r, 'std_g': std_g, 'std_b': std_b}

    # Grayscale image
    else:
        value_ = round(channel_mean[0], 2)
        std = round(channel_std[0], 2)

        return {'value': value_, 'std': std}


# 전체 학습 데이터셋으로부터 sub dataset 생성
# Create Date : 2026.03.19
# Last Update Date : -

# Arguments:
# - dataset_name         (str)                  : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - avg_pixel_hsv_ranges (list([float, float])) : 특정 이미지의 모든 픽셀을 channel-wise 하게 평균한 픽셀의 색상,채도,명도 값 범위
#                                                 (색상, 채도, 명도 각각 sub-list)
# - std_ranges           (list([float, float])) : 특정 이미지의 모든 픽셀에 대한 R, G, B 값 각각의 표준편차 범위
#                                                 (R, G, B 각각 sub-list)
# - save_dir             (str)                  : sub-dataset 저장 경로

# Returns
# - train_subdataset_count (dict) : 학습 데이터의 subdataset 에서 각 class 별 이미지 개수
# - test_subdataset_count  (dict) : 테스트 데이터의 subdataset 에서 각 class 별 이미지 개수

def create_subdataset(dataset_name, avg_pixel_hsv_ranges, std_ranges, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    dataset_dir = f'{PROJECT_DIR_PATH}/datasets/{dataset_name}'
    class_names = os.listdir(os.path.join(dataset_dir, 'train'))

    if dataset_name == 'cifar_10':  # RGB image
        dataset_hsv_std_info_dict = {'tvt_type': [], 'img_path': [], 'label': [],
                                     'hue': [], 'saturation': [], 'value': [],
                                     'std_r': [], 'std_g': [], 'std_b': []}

    else:  # Grayscale image
        dataset_hsv_std_info_dict = {'tvt_type': [], 'img_path': [], 'label': [],
                                     'value': [], 'std': []}

    train_subdataset_count = {}
    test_subdataset_count = {}

    for class_name in class_names:
        train_subdataset_count[class_name] = 0
        test_subdataset_count[class_name] = 0

        train_save_dir_for_class = os.path.join(save_dir, 'train', class_name)
        test_save_dir_for_class = os.path.join(save_dir, 'test', class_name)
        os.makedirs(train_save_dir_for_class, exist_ok=True)
        os.makedirs(test_save_dir_for_class, exist_ok=True)

        train_img_dir_name = f'{dataset_dir}/train/{class_name}'
        test_img_dir_name = f'{dataset_dir}/test/{class_name}'
        train_img_names = os.listdir(train_img_dir_name)
        test_img_names = os.listdir(test_img_dir_name)

        print(f'processing class : {class_name} (train: {len(train_img_names)}, test: {len(test_img_names)})')

        # generate sub-dataset (train)
        for train_img_name in train_img_names:
            train_img_path = os.path.join(train_img_dir_name, train_img_name)
            train_img_save_path = os.path.join(train_save_dir_for_class, train_img_name)

            if dataset_name == 'cifar_10':
                train_img = cv2.imread(train_img_path)
            else:
                train_img = cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE)
                train_img = train_img[:, :, None]

            # check condition and save
            is_saved = check_condition_and_save_image_in_subdataset(img=train_img,
                                                                    avg_pixel_hsv_ranges=avg_pixel_hsv_ranges,
                                                                    std_ranges=std_ranges,
                                                                    save_path=train_img_save_path)

            if is_saved:
                train_subdataset_count[class_name] += 1
                if train_subdataset_count[class_name] >= MAX_DATA_SIZE_PER_CLASS:
                    break

            # add to dataset info dictionary
            dataset_hsv_std_info_dict['tvt_type'].append('train')
            dataset_hsv_std_info_dict['img_path'].append(f'{class_name}/{train_img_name}')
            dataset_hsv_std_info_dict['label'].append(class_name)

            image_condition_values = compute_image_condition_values(train_img)
            for k, v in image_condition_values.items():
                dataset_hsv_std_info_dict[k].append(v)

        # generate sub-dataset (test)
        for test_img_name in test_img_names:
            test_img_path = os.path.join(test_img_dir_name, test_img_name)
            test_img_save_path = os.path.join(test_save_dir_for_class, test_img_name)

            if dataset_name == 'cifar_10':
                test_img = cv2.imread(test_img_path)
            else:
                test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
                test_img = test_img[:, :, None]

            # check condition and save
            is_saved = check_condition_and_save_image_in_subdataset(img=test_img,
                                                                    avg_pixel_hsv_ranges=avg_pixel_hsv_ranges,
                                                                    std_ranges=std_ranges,
                                                                    save_path=test_img_save_path)

            if is_saved:
                test_subdataset_count[class_name] += 1
                if test_subdataset_count[class_name] >= MAX_DATA_SIZE_PER_CLASS:
                    break

            # add to dataset info dictionary
            dataset_hsv_std_info_dict['tvt_type'].append('test')
            dataset_hsv_std_info_dict['img_path'].append(f'{class_name}/{test_img_name}')
            dataset_hsv_std_info_dict['label'].append(class_name)

            image_condition_values = compute_image_condition_values(test_img)
            for k, v in image_condition_values.items():
                dataset_hsv_std_info_dict[k].append(v)

    # convert to Pandas DataFrame and save
    dataset_hsv_std_info_df = pd.DataFrame(dataset_hsv_std_info_dict)
    dataset_hsv_std_info_df_path = os.path.join(save_dir, 'dataset_info.csv')
    dataset_hsv_std_info_df.to_csv(dataset_hsv_std_info_df_path)

    return train_subdataset_count, test_subdataset_count


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        print(f'\n==== DATASET: {dataset_name} ====\n')

        if dataset_name == 'cifar_10':
            avg_pixel_hsv_ranges = [[0, 255], [0, 255], [224, 255]]
            std_ranges = [[0, 40], [0, 40], [0, 40]]
        else:
            avg_pixel_hsv_ranges = [[48, 64]]
            std_ranges = [[0, 255]]

        save_dir = f'{PROJECT_DIR_PATH}/hpo_training_data/test/{dataset_name}'
        train_subdataset_count, test_subdataset_count = create_subdataset(dataset_name,
                                                                          avg_pixel_hsv_ranges,
                                                                          std_ranges,
                                                                          save_dir)

        print(f'TRAIN subdataset image count per class:\n{train_subdataset_count}')
        print(f'TEST subdataset image count per class:\n{test_subdataset_count}')
