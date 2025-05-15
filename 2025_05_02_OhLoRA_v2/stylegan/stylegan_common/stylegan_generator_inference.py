# Modified implementation from https://github.com/genforce/genforce/blob/master/runners/base_gan_runner.py

import time
import os
import numpy as np
import torch
# import torch.distributed as dist

try:
    from stylegan_common.visualizer import postprocess_image, save_image
except:
    from stylegan.stylegan_common.visualizer import postprocess_image, save_image


PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

RANK = 0
WORLD_SIZE = 1
VALID_BATCH_SIZE = 4
IMG_RESOLUTION = 256
DIM_W = 512


def synthesize(generator_model, num, save_dir, z=None, label=None, img_name_start_idx=0, verbose=True,
               save_img=True, return_img=False, return_w=False):
    """Synthesizes images.

    Args:
        generator_model: GAN Generator Model.
        num: Number of images to synthesize.
        save_dir: GAN-generated image save path.
        z: Latent codes used for generation. If not specified, this function
            will sample latent codes randomly. (default: None)
        label: additional label for conditional generation. (default: None)
        img_name_start_idx: start index number for image name.
        verbose: whether to print info
        save_img: whether to save synthesized images
        return_img: whether to return synthesized images
        return_w: whether to return w (intermediate latent vectors)
    """

    os.makedirs(save_dir, exist_ok=True)

    if z is not None:
        assert isinstance(z, np.ndarray)
        assert z.ndim == 2 and z.shape[1] == generator_model.z_space_dim
        num = min(num, z.shape[0])
        z = torch.from_numpy(z).type(torch.FloatTensor)
    if label is not None:
        assert isinstance(label, np.ndarray)
        assert label.ndim == 2 and label.shape[1] == generator_model.label_size
        label = torch.from_numpy(label).type(torch.FloatTensor)
    if not num:
        return
    # TODO: Use same z during the entire training process.

    indices = list(range(RANK, num, WORLD_SIZE))
    batch_count = len(indices) // VALID_BATCH_SIZE
    start_at = time.time()
    all_images = np.zeros((num, IMG_RESOLUTION, IMG_RESOLUTION, 3))
    all_ws = np.zeros((num, DIM_W))

    for batch_idx in range(0, len(indices), VALID_BATCH_SIZE):
        sub_indices = indices[batch_idx:batch_idx + VALID_BATCH_SIZE]
        batch_size = len(sub_indices)

        if z is None:
            code = torch.randn(batch_size, generator_model.z_space_dim).cuda()
        else:
            code = z[sub_indices].cuda()

        if label is None:
            property_vector = torch.randn(batch_size, generator_model.label_size).cuda()
        else:
            property_vector = label[sub_indices].cuda()

        with torch.no_grad():
            if return_w:
                gen_results = generator_model(code, property_vector, **generator_model.G_kwargs_val)
                images = gen_results['image']
                images = postprocess_image(images.detach().cpu().numpy())
                ws = gen_results['w'].detach().cpu().numpy()
            else:
                images = generator_model(code, property_vector, **generator_model.G_kwargs_val)['image']
                images = postprocess_image(images.detach().cpu().numpy())

        if save_img:
            for sub_idx, image in zip(sub_indices, images):
                save_image(os.path.join(save_dir, f'{sub_idx+img_name_start_idx:06d}.jpg'), image)

        elapsed_time = time.time() - start_at
        img_cnt = batch_idx + VALID_BATCH_SIZE
        avg_time = elapsed_time / img_cnt

        if verbose and (batch_idx < 100 or batch_idx % 100 == 0):
            print(f'image {img_cnt} / {num}, time : {elapsed_time:.4f}, time/image : {avg_time:.4f}')

        if return_img:
            all_images[batch_idx:batch_idx + batch_size] = images

        if return_w:
            all_ws[batch_idx:batch_idx + batch_size] = ws

    if return_img and return_w:
        return all_images, all_ws
    elif return_w:
        return all_ws
    elif return_img:
        return all_images

#    dist.barrier()
