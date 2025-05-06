from stylegan_vectorfind_v6.main import main as stylegan_vectorfind_v6_main
from common import load_existing_stylegan_finetune_v1
import stylegan_common.stylegan_generator as gen

import torch
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v1 : {device}')

    finetune_v1_generator = gen.StyleGANGeneratorForV6(resolution=IMAGE_RESOLUTION)
    generator_state_dict = load_existing_stylegan_finetune_v1(device)
    print('Existing StyleGAN-FineTune-v1 Generator load successful!! 😊')

    # load state dict (generator)
    del generator_state_dict['mapping.label_weight']  # size mismatch because of modified property vector dim (7 -> 3)
    finetune_v1_generator.load_state_dict(generator_state_dict, strict=False)

    # get property score changing vector
    stylegan_vectorfind_v6_main(finetune_v1_generator)

    # image generation test
    # TODO: implement

