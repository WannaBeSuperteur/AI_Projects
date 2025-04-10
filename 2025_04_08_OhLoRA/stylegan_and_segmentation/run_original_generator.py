import stylegan.stylegan_generator as gen
import stylegan.stylegan_generator_inference as inference
import torch

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for running StyleGAN Generator : {device}')

    # load generator state dict
    generator_model = gen.StyleGANGenerator(resolution=256)
    model_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan/stylegan_model.pth'

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    generator_state_dict = state_dict['generator']

    generator_model.load_state_dict(generator_state_dict)
    generator_model.to(device)

    # run generator
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    generator_model.G_kwargs_val = kwargs_val

    inference.synthesize(generator_model, num=100, z=None)
