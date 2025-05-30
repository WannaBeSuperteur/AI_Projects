import torch
import torchvision.transforms as transforms
from torchvision.io import read_image

import pandas as pd

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

import stylegan.stylegan_common.stylegan_generator as gen
from stylegan.stylegan_common.visualizer import save_image, postprocess_image
from property_score_cnn.run_merged_cnn import MergedPropertyScoreCNN


IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS = 7

MERGED_PROPERTY_SCORE_CNN_PATH = f'{PROJECT_DIR_PATH}/property_score_cnn/models/ohlora_v3_merged_property_cnn.pth'
GENERATED_IMG_PATH = f'{PROJECT_DIR_PATH}/v8_property_scores/generated_images'
os.makedirs(GENERATED_IMG_PATH, exist_ok=True)
kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)

stylegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


# StyleGAN-FineTune-v8 모델의 Generator 로딩
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - finetune_v8_generator (nn.Module) : StyleGAN-FineTune-v8 모델의 Generator

def load_stylegan_finetune_v8_generator(device):
    generator_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v8.pth'
    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)

    finetune_v8_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)
    finetune_v8_generator.load_state_dict(generator_state_dict)
    finetune_v8_generator.to(device)
    print('Existing StyleGAN-FineTune-v8 Generator load successful!! 😊')

    return finetune_v8_generator


# Merged Property Score CNN (hairstyle 포함한 핵심 속성 값 계산용 CNN) 로딩
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - device (Device) : CUDA or CPU device

# Returns:
# - merged_property_score_cnn (nn.Module) : Merged Property Score CNN (핵심 속성 값 계산용 CNN)

def load_merged_property_score_cnn(device):
    merged_property_cnn_model = MergedPropertyScoreCNN()
    merged_property_cnn_state_dict = torch.load(MERGED_PROPERTY_SCORE_CNN_PATH,
                                                map_location=device,
                                                weights_only=False)
    merged_property_cnn_model.load_state_dict(merged_property_cnn_state_dict)

    merged_property_cnn_model.to(device)
    merged_property_cnn_model.device = device
    print('Existing Merged Property CNN load successful!! 😊')

    return merged_property_cnn_model


# StyleGAN-FineTune-v8 모델의 Generator 를 이용하여 이미지 생성
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - finetune_v8_generator (nn.Module) : StyleGAN-FineTune-v8 모델의 Generator
# - n                     (int)       : 생성할 이미지 개수

def generate_images(finetune_v8_generator, n=20000):
    for i in range(n):
        if i % 200 == 0:
            print(f'generating : {i} / {n}')

        with torch.no_grad():
            code_part1 = torch.randn(1, ORIGINAL_HIDDEN_DIMS_Z).type(torch.float32)
            code_part2 = torch.randn(1, ORIGINALLY_PROPERTY_DIMS).type(torch.float32)

            images = finetune_v8_generator(code_part1.cuda(), code_part2.cuda(), **kwargs_val)['image']
            images = postprocess_image(images.detach().cpu().numpy())

            save_image(os.path.join(GENERATED_IMG_PATH, f'{i:06d}.jpg'), images[0])


# 생성된 이미지에 대해 핵심 속성 값 (hair_color, hair_length, background_score, hairstyle) 도출 및 그 결과를 csv로 저장
# Create Date : 2025.05.29
# Last Update Date : -

# Arguments:
# - merged_property_score_cnn (nn.Module) : Merged Property Score CNN (핵심 속성 값 계산용 CNN)

def compute_property_scores(merged_property_score_cnn):
    image_cnt = len(list(filter(lambda x: x.endswith('.jpg'), os.listdir(GENERATED_IMG_PATH))))

    score_dict = {'img_no': list(range(image_cnt)),
                  'hair_color': [], 'hair_length': [], 'background_score': [], 'hairstyle': []}

    mean_and_median_dict = {'value_type': ['mean', 'median'],
                            'hair_color': [], 'hair_length': [], 'background_score': [], 'hairstyle': []}

    # compute scores
    with torch.no_grad():
        for img_no in range(image_cnt):
            image_path = f'{GENERATED_IMG_PATH}/{img_no:06d}.jpg'
            image = read_image(image_path)
            image = stylegan_transform(image)
            image = image.unsqueeze(0)

            output_scores = merged_property_score_cnn(image.cuda()).detach().cpu().numpy()
            score_dict['hair_color'].append(round(output_scores[0][1], 4))
            score_dict['hair_length'].append(round(output_scores[0][2], 4))
            score_dict['background_score'].append(round(output_scores[0][5], 4))
            score_dict['hairstyle'].append(round(output_scores[0][7], 4))

    score_df = pd.DataFrame(score_dict)
    score_df.to_csv(f'{PROJECT_DIR_PATH}/v8_property_scores/property_scores_sampled.csv',
                    index=False)

    # compute mean and median of scores
    for property in ['hair_color', 'hair_length', 'background_score', 'hairstyle']:
        mean_and_median_dict[property].append(round(score_df[property].mean(), 4))
        mean_and_median_dict[property].append(round(score_df[property].median(), 4))

    mean_and_median_df = pd.DataFrame(mean_and_median_dict)
    mean_and_median_df.to_csv(f'{PROJECT_DIR_PATH}/v8_property_scores/property_scores_mean_and_median.csv',
                              index=False)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for generating image using StyleGAN-FineTune-v8 : {device}')

    finetune_v8_generator = load_stylegan_finetune_v8_generator(device)
    merged_property_score_cnn = load_merged_property_score_cnn(device)

    generate_images(finetune_v8_generator)
    compute_property_scores(merged_property_score_cnn)
