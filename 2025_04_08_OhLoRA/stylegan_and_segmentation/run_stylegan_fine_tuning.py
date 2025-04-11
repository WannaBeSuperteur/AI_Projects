import stylegan.stylegan_generator as gen
import stylegan.stylegan_discriminator as dis
import torch

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# 기존 Pre-train 된 StyleGAN 모델 로딩
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - pretrained_generator     (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator
# - pretrained_discriminator (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator

def load_existing_stylegan():
    pretrained_generator = gen.StyleGANGenerator(resolution=256)
    pretrained_discriminator = dis.StyleGANDiscriminator(resolution=256)

    model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/stylegan_model.pth'

    # load generator state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    generator_state_dict = state_dict['generator']
    discriminator_state_dict = state_dict['discriminator']

    pretrained_generator.load_state_dict(generator_state_dict)
    pretrained_discriminator.load_state_dict(discriminator_state_dict)

    pretrained_generator.to(device)
    pretrained_discriminator.to(device)

    return pretrained_generator, pretrained_discriminator


# StyleGAN Fine-Tuning 실시 (핵심 속성 값 5개를 latent vector 에 추가)
# Create Date : 2025.04.11
# Last Update Date : -

# Arguments:
# - pretrained_generator     (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator
# - pretrained_discriminator (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Discriminator

# Returns:
# - stylegan_modified/stylegan_model_fine_tuned.pth 에 Fine-Tuning 된 StyleGAN 저장

def run_fine_tuning(pretrained_generator, pretrained_discriminator):
    raise NotImplementedError


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training StyleGAN : {device}')

    # load Pre-trained StyleGAN
    pretrained_gen, pretrained_dis = load_existing_stylegan()

    # Fine Tuning
    run_fine_tuning(pretrained_gen, pretrained_dis)

