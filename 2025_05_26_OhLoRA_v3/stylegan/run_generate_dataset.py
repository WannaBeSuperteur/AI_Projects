import stylegan_common.stylegan_generator as gen
from common import load_existing_stylegan_finetune_v1, save_model_structure_pdf
from generate_dataset.generate import generate_face_images

import torch
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

IMAGE_RESOLUTION = 256
PDF_BATCH_SIZE = 30
ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS = 7


# StyleGAN-FineTune-v1 Generator 의 구조를 PDF 파일로 저장
# Create Date : 2025.05.26
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

# Returns:
# - stylegan/model_structure_pdf 에 StyleGAN-FineTune-v1 generator 구조를 나타내는 PDF 파일 저장

def create_model_structure_pdf(finetune_v1_generator):
    save_model_structure_pdf(finetune_v1_generator,
                             model_name='finetune_v1_generator_for_train_data_generation',
                             input_size=[(PDF_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (PDF_BATCH_SIZE, ORIGINALLY_PROPERTY_DIMS)],
                             print_frozen=False)


if __name__ == '__main__':
    fine_tuned_model_path = f'{PROJECT_DIR_PATH}/stylegan/models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for inferencing StyleGAN-FineTune-v1 : {device}')

    finetune_v1_generator = gen.StyleGANGenerator(resolution=IMAGE_RESOLUTION)

    # try loading StyleGAN-FineTune-v1 pre-trained model
    generator_state_dict = load_existing_stylegan_finetune_v1(device)
    finetune_v1_generator.load_state_dict(generator_state_dict)
    print('Existing StyleGAN-FineTune-v1 Generator load successful!! 😊')

    # create model structure PDF and save
    finetune_v1_generator.to(device)
    create_model_structure_pdf(finetune_v1_generator)

    # image generation test
    generate_face_images(finetune_v1_generator)
