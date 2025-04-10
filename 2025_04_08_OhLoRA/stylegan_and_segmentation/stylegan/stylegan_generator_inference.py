# Modified implementation from https://github.com/genforce/genforce/blob/master/runners/base_gan_runner.py

import os
import numpy as np
import torch
# import torch.distributed as dist

from stylegan.visualizer import postprocess_image, save_image

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

RANK = 0
WORLD_SIZE = 1
VALID_BATCH_SIZE = 4


def synthesize(generator_model, num, z=None):
    """Synthesizes images.

    Args:
        generator_model: GAN Generator Model.
        num: Number of images to synthesize.
        z: Latent codes used for generation. If not specified, this function
            will sample latent codes randomly. (default: None)
    """

    temp_dir = f'{PROJECT_DIR_PATH}/stylegan/synthesize_results'
    os.makedirs(temp_dir, exist_ok=True)

    if z is not None:
        assert isinstance(z, np.ndarray)
        assert z.ndim == 2 and z.shape[1] == generator_model.z_space_dim
        num = min(num, z.shape[0])
        z = torch.from_numpy(z).type(torch.FloatTensor)
    if not num:
        return
    # TODO: Use same z during the entire training process.

    indices = list(range(RANK, num, WORLD_SIZE))
    for batch_idx in range(0, len(indices), VALID_BATCH_SIZE):
        sub_indices = indices[batch_idx:batch_idx + VALID_BATCH_SIZE]
        batch_size = len(sub_indices)
        if z is None:
            code = torch.randn(batch_size, generator_model.z_space_dim).cuda()
        else:
            code = z[sub_indices].cuda()
        with torch.no_grad():
            images = generator_model(code, **generator_model.G_kwargs_val)['image']
            images = postprocess_image(images.detach().cpu().numpy())
        for sub_idx, image in zip(sub_indices, images):
            save_image(os.path.join(temp_dir, f'{sub_idx:06d}.jpg'), image)

#    dist.barrier()
