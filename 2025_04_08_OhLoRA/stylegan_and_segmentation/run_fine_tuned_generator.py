import stylegan_modified.stylegan_generator as gen
import stylegan_modified.stylegan_generator_inference as inference

import torch
import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

TEST_GEN_IMAGE_COUNT = 20
ORIGINAL_HIDDEN_DIMS_Z = 512


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for running StyleGAN Generator : {device}')

    # load generator state dict
    generator_model = gen.StyleGANGeneratorForV3(resolution=256)
    stylegan_modified_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    generator_path = f'{stylegan_modified_dir}/stylegan_gen_fine_tuned_v3_ckpt_0005_gen.pth'

    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    generator_model.load_state_dict(generator_state_dict)
    generator_model.to(device)

    # run testing generator
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    generator_model.G_kwargs_val = kwargs_val

    inference_save_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/final_inference_test_v3'
    z_vector_csv_path = f'{stylegan_modified_dir}/test_z_vector_0005.csv'
    z_vector = pd.read_csv(z_vector_csv_path)
    z_vector_np = np.array(z_vector)
    label_np = np.array(TEST_GEN_IMAGE_COUNT * [[1.8, 0.0, 0.0, 1.6, -1.2, 0.0, 0.0]])

    for i in range(TEST_GEN_IMAGE_COUNT - 1):
        z_vector_with_noise = z_vector_np[:1] + 0.3 * np.random.randn(1, ORIGINAL_HIDDEN_DIMS_Z)
        z_vector_np = np.concatenate([z_vector_np, z_vector_with_noise], axis=0)

    inference.synthesize(generator_model,
                         save_dir=inference_save_dir,
                         num=TEST_GEN_IMAGE_COUNT,
                         z=z_vector_np,
                         label=label_np)
