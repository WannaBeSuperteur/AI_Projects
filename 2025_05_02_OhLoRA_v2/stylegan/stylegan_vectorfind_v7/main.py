try:
    import stylegan_common.stylegan_generator_inference as infer
    from stylegan_vectorfind_v7.run_vector_find import run_stylegan_vector_find
    from common import save_model_structure_pdf
except:
    import stylegan.stylegan_common.stylegan_generator_inference as infer
    from stylegan.stylegan_vectorfind_v7.run_vector_find import run_stylegan_vector_find
    from stylegan.common import save_model_structure_pdf

import os


PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/stylegan/model_structure_pdf'

os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)


IMAGE_RESOLUTION = 256
ORIGINAL_HIDDEN_DIMS_Z = 512
ORIGINALLY_PROPERTY_DIMS_Z = 3  # 원래 property (eyes, mouth, pose) 목적으로 사용된 dimension 값
PDF_BATCH_SIZE = 30


# StyleGAN-FineTune-v1 Generator 의 구조를 PDF 파일로 저장
# Create Date : 2025.05.15
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

# Returns:
# - stylegan/model_structure_pdf 에 StyleGAN-FineTune-v1 generator 구조를 나타내는 PDF 파일 저장

def create_model_structure_pdf(finetune_v1_generator):
    save_model_structure_pdf(finetune_v1_generator,
                             model_name='finetune_v1_generator_for_v7',
                             input_size=[(PDF_BATCH_SIZE, ORIGINAL_HIDDEN_DIMS_Z),
                                         (PDF_BATCH_SIZE, ORIGINALLY_PROPERTY_DIMS_Z)],
                             print_frozen=False)


# StyleGAN Vector Finding 이전 inference test 실시
# Create Date : 2025.05.15
# Last Update Date : -

# Arguments:
# - finetune_v1_generator (nn.Module) : StyleGAN-FineTune-v1 의 Generator

# Returns:
# - stylegan/stylegan_vectorfind_v7/inference_test_before_finetuning 에 생성 결과 저장

def run_inference_test_before_training(finetune_v1_generator):
    kwargs_val = dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False)
    finetune_v1_generator.G_kwargs_val = kwargs_val

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/inference_test_before_training'
    infer.synthesize(finetune_v1_generator, num=50, save_dir=img_save_dir, z=None, label=None)


def main(finetune_v1_generator, device):

    # model structure PDF file
    create_model_structure_pdf(finetune_v1_generator)

    # run inference test before training
    run_inference_test_before_training(finetune_v1_generator)

    # Fine Tuning
    run_stylegan_vector_find(finetune_v1_generator, device)
