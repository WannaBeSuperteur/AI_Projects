from stylegan_common.visualizer import save_image, postprocess_image

import torch
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS = 7
kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)


# StyleGAN-FineTune-v8 & StyleGAN-VectorFind-v8 학습 데이터 용 얼굴 이미지 생성
# Create Date : 2025.05.26
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator]
# - n                     (int)       : 생성할 이미지의 개수 (default : 15,000)

def generate_face_images(finetune_v1_generator, n=15000):
    save_dir = f'{PROJECT_DIR_PATH}/stylegan/generated_face_images'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n):
        if i % 100 == 0:
            print(f'generating : {i} / {n}')

        with torch.no_grad():
            code_part1 = torch.randn(1, ORIGINAL_HIDDEN_DIMS_Z).type(torch.float32)
            code_part2 = torch.randn(1, ORIGINALLY_PROPERTY_DIMS).type(torch.float32)

            images = finetune_v1_generator(code_part1.cuda(), code_part2.cuda(), **kwargs_val)['image']
            images = postprocess_image(images.detach().cpu().numpy())

            save_image(os.path.join(save_dir, f'{i:06d}.jpg'), images[0])
