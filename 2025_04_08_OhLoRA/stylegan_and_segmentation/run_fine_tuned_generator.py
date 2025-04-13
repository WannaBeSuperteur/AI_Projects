import stylegan_modified.stylegan_generator as gen  # TODO update import path
import stylegan_modified.stylegan_generator_inference as inference  # TODO update import path
import torch

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


if __name__ == '__main__':

    # version No.
    v = 1

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for running StyleGAN Generator : {device}')

    # load generator state dict
    generator_model = gen.StyleGANGenerator(resolution=256)
    generator_path = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/stylegan_gen_fine_tuned_v{v}.pth'

    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    generator_model.load_state_dict(generator_state_dict)
    generator_model.to(device)

    # run generator
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    generator_model.G_kwargs_val = kwargs_val

    inference_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/inference_test_v{v}'
    inference.synthesize(generator_model, save_dir=inference_save_dir, num=300, z=None)
