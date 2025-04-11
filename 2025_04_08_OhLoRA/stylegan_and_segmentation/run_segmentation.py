# Original Pre-trained FaceXFormer from https://huggingface.co/kartiknarayan/facexformer/tree/main/ckpts > model.pt


from segmentation.inference import test

import torch
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Model 을 이용하여 Face Segmentation 실시
# Create Date : 2025.04.10
# Last Update Date : 2025.04.11
# - inferencing image count 추가

# Arguments:
# - img_paths  (list(str)) : Face Segmentation 을 실시할 이미지의 경로 목록
# - model_path (str)       : Segmentation 용 Pre-trained Model 의 path

# Outputs:
# - segmentation/segmentation_results 디렉토리에 Face Segmentation 결과 저장

def run_segmentation_with_model(img_paths, model_path):
    result_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/segmentation_results'
    os.makedirs(result_path, exist_ok=True)

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training : {device}')

    gpu_num = '0' if 'cuda' in str(device) else 'cpu'

    for idx, img_path in enumerate(img_paths):
        if idx < 10 or idx % 25 == 0:
            print(f'inferencing image {idx} ...')

        img_idx = int(img_path.split('/')[-1][:-4])  # image index from ORIGINAL DATASET with 10K generated images

        test_args = {'model_path': model_path,
                     'image_path': img_path,
                     'results_path': result_path,
                     'task': 'parsing',
                     'gpu_num': gpu_num,
                     'img_idx': img_idx}

        test(test_args)


if __name__ == '__main__':

    # model path
    model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/segmentation/models/segmentation_model.pt'

    # get image path
    img_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/synthesize_results_filtered'
    img_names = os.listdir(img_dir)
    img_paths = [f'{img_dir}/{img_name}' for img_name in img_names]

    # run segmentation
    run_segmentation_with_model(img_paths, model_path)
