
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# 전체 학습 데이터셋으로부터 sub dataset 생성
# Create Date : 2026.03.19
# Last Update Date : -

# Arguments:
# - dataset_name     (str)                  : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')
# - avg_pixel_ranges (list([float, float])) : 특정 이미지의 모든 픽셀을 평균한 픽셀의 색상, 채도, 명도 값의 범위 (색상, 채도, 명도 각각 sub-list)
# - std_ranges       (list([float, float])) : 특정 이미지의 모든 픽셀에 대한 R, G, B 값 각각의 표준편차 범위 (R, G, B 각각 sub-list)
# - save_dir         (str)                  : sub-dataset 저장 경로

def create_subdataset(dataset_name, avg_pixel_ranges, std_ranges, save_dir):
    raise NotImplementedError


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        if dataset_name == 'cifar_10':
            avg_pixel_ranges = [[80, 120], [0, 255], [0, 255]]
            std_ranges = [[0, 40], [0, 40], [0, 40]]
        else:
            avg_pixel_ranges = [[64, 76]]
            std_ranges = [[0, 255]]

        save_dir = f'{PROJECT_DIR_PATH}/hpo_training_data/test/{dataset_name}'
        os.makedirs(save_dir, exist_ok=True)
        create_subdataset(dataset_name, avg_pixel_ranges, std_ranges, save_dir)
