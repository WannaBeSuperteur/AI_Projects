import torch
import torch.nn as nn
from torchview import draw_graph
import pandas as pd

from stylegan_modified.stylegan_generator import StyleGANGenerator
from stylegan_modified.stylegan_generator_v2_cnn import PropertyScoreCNN

import os
import sys

global_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
sys.path.append(global_path)

from global_common.visualize_tensor import save_tensor_png


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/model_structure_pdf'

TENSOR_VISUALIZE_TEST_BATCH_SIZE = 30
TRAIN_BATCH_SIZE = 16

IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 7           # eyes, hair_color, hair_length, mouth, pose, background_mean, background_std

CNN_TENSOR_TEST_DIR = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified/tensor_visualize_test_cnn'
os.makedirs(CNN_TENSOR_TEST_DIR, exist_ok=True)


class StyleGANFineTuneV2(nn.Module):
    def __init__(self):
        super(StyleGANFineTuneV2, self).__init__()

        self.stylegan_generator = StyleGANGenerator(resolution=IMAGE_RESOLUTION)
        self.property_score_cnn = PropertyScoreCNN()

        kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
        self.stylegan_generator.G_kwargs_val = kwargs_val

    def forward(self, z, property_label, tensor_visualize_test=False):
        generated_image = self.stylegan_generator(z, property_label)
        property_score = self.property_score_cnn(generated_image['image'])

        if tensor_visualize_test:
            test_name = 'test_before_finetune'

            current_batch_size = generated_image['image'].size(0)
            property_score_np = property_score.detach().cpu().numpy()

            property_score_info_dict = {
                'img_no': list(range(current_batch_size)),
                'eyes_score': list(property_score_np[:, 0]),
                'hair_color_score': list(property_score_np[:, 1]),
                'hair_length_score': list(property_score_np[:, 2]),
                'mouth_score': list(property_score_np[:, 3]),
                'pose_score': list(property_score_np[:, 4]),
                'back_mean_score': list(property_score_np[:, 5]),
                'back_std_score': list(property_score_np[:, 6])
            }
            property_score_info_df = pd.DataFrame(property_score_info_dict)
            property_score_info_df.to_csv(f'{CNN_TENSOR_TEST_DIR}/finetune_v2_{test_name}_result.csv',
                                          index=False)

            for i in range(current_batch_size):
                save_tensor_png(generated_image['image'][i],
                                image_save_path=f'{CNN_TENSOR_TEST_DIR}/finetune_v2_{test_name}_{i:03d}.png')

        return property_score


# StyleGAN-FineTune-v2 모델 정의 및 generator 의 state_dict 를 로딩
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - device    (device)    : 모델을 mapping 시킬 device (GPU 등)
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - cnn_model (nn.Module) : StyleGAN-FineTune-v2 Fine-Tuning 에 사용할 학습된 CNN 모델

# Returns:
# - stylegan_finetune_v2 (nn.Module) : 학습할 StyleGAN-FineTune-v2 모델

def define_stylegan_finetune_v2(device, generator, cnn_model):
    stylegan_finetune_v2 = StyleGANFineTuneV2()
    stylegan_finetune_v2.optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.00005)
    stylegan_finetune_v2.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=stylegan_finetune_v2.optimizer,
                                                                                T_max=10,
                                                                                eta_min=0)

    stylegan_finetune_v2.to(device)
    stylegan_finetune_v2.device = device

    # load state dict of generator and CNN model
    stylegan_finetune_v2.stylegan_generator.load_state_dict(generator.state_dict())
    stylegan_finetune_v2.property_score_cnn.load_state_dict(cnn_model.state_dict())

    # save model graph of StyleGAN-FineTune-v2 before training
    model_graph = draw_graph(stylegan_finetune_v2,
                             input_data=[torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z)),
                                         torch.randn((TENSOR_VISUALIZE_TEST_BATCH_SIZE, PROPERTY_DIMS_Z))],
                             depth=5)

    visual_graph = model_graph.visual_graph

    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/stylegan_finetune_v2.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    return stylegan_finetune_v2


# 정의된 StyleGAN-FineTune-v2 모델을 학습
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - stylegan_finetune_v2 (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator
#                                      (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)

def run_training_stylegan_finetune_v2(stylegan_finetune_v2):
    raise NotImplementedError


# StyleGAN-FineTune-v2 모델 학습
# Create Date : 2025.04.14
# Last Update Date : -

# Arguments:
# - device    (device)    : 모델을 mapping 시킬 device (GPU 등)
# - generator (nn.Module) : StyleGAN-FineTune-v1 모델의 Generator (Fine-Tuning 대상)
# - cnn_model (nn.Module) : StyleGAN-FineTune-v2 Fine-Tuning 에 사용할 학습된 CNN 모델

# Returns:
# - fine_tuned_generator (nn.Module) : Fine-Tuning 된 StyleGAN-FineTune-v2 모델의 Generator
#                                      (StyleGAN-FineTune-v1 모델을 Fine-Tuning 시킨)

def train_stylegan_finetune_v2(device, generator, cnn_model):
    stylegan_finetune_v2 = define_stylegan_finetune_v2(device, generator, cnn_model)
    fine_tuned_generator = run_training_stylegan_finetune_v2(stylegan_finetune_v2)

    return fine_tuned_generator
