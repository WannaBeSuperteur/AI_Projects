import stylegan_modified.stylegan_generator as gen  # TODO update import path
import stylegan_modified.stylegan_generator_inference as inference  # TODO update import path
import torch

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for running StyleGAN Generator : {device}')

    # load generator state dict
    generator_model = gen.StyleGANGenerator(resolution=256)
    generator_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned.pth'

    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    generator_model.load_state_dict(generator_state_dict)
    generator_model.to(device)

    # run generator
    inference.synthesize(generator_model, num=200, z=None)
