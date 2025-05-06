from stylegan_finetune_v5.main import main as stylegan_finetune_v5_main
from stylegan_finetune_v5.run_fine_tuning import run_inference_test
from property_score_cnn import load_cnn_model as load_property_cnn_model
import stylegan_common.stylegan_generator as gen

import torch
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
IMAGE_RESOLUTION = 256


# Fine-Tune 된 StyleGAN-FineTune-v5 모델 로딩
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - device (device) : StyleGAN-FineTune-v4 모델을 mapping 시킬 device (GPU 등)

# Returns:
# - finetune_v5_generator (nn.Module) : StyleGAN-FineTune-v5 모델의 Generator

def load_existing_stylegan(device):
    finetune_v5_generator = gen.StyleGANGeneratorForV5(resolution=IMAGE_RESOLUTION)

    # load generator state dict
    gen_model_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/stylegan_gen_fine_tuned_v5.pth'
    generator_state_dict = torch.load(gen_model_path, map_location=device, weights_only=True)

    del generator_state_dict['truncation.w_avg']  # size mismatch because of conditional truncation trick
    finetune_v5_generator.load_state_dict(generator_state_dict, strict=False)
    finetune_v5_generator.to(device)

    return finetune_v5_generator


# StyleGAN Fine-Tuning 이후 inference test 실시
# Create Date : 2025.05.06
# Last Update Date : -

# Arguments:
# - finetune_v5_generator (nn.Module) : StyleGAN-FineTune-v5 모델의 Generator
# - trunc_psi             (float)     : StyleGAN-FineTune-v5 모델에 적용되는 psi 값 (0 에 가까울수록 w 를 center mass 로 더 많이 이동)
# - epoch_no_temp         (int)       : 여러 가지 trunc_psi 값으로 생성된 이미지의 경로를 구분하기 위한 epoch 번호

# Returns:
# - stylegan/stylegan_finetune_v5/inference_test_before_finetuning 에 생성 결과 저장

def run_inference_test_after_finetuning(finetune_v5_generator, trunc_psi=0.5, epoch_no_temp=0):
    kwargs_val = dict(trunc_psi=trunc_psi, trunc_layers=0, randomize_noise=False)
    finetune_v5_generator.G_kwargs_val = kwargs_val

    property_cnn_path = f'{PROJECT_DIR_PATH}/stylegan/models/stylegan_gen_fine_tuned_v2_cnn.pth'
    property_score_cnn = load_property_cnn_model(property_cnn_path, device)

    img_save_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_finetune_v5/inference_test_after_finetuning'
    inference_test_result = run_inference_test(finetune_v5_generator,
                                               img_save_dir,
                                               current_epoch=epoch_no_temp,
                                               batch_idx=0,
                                               property_score_cnn=property_score_cnn)

    print(f'inference_test_result :\n{inference_test_result}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device for training / inferencing StyleGAN-FineTune-v5 : {device}')

    try:
        finetune_v5_generator = load_existing_stylegan(device)
        print('Existing StyleGAN-FineTune-v5 load successful!! 😊')

    except Exception as e:
        print(f'Existing StyleGAN-FineTune-v5 load failed : {e}')
        stylegan_finetune_v5_main()

        finetune_v5_generator = load_existing_stylegan(device)
        print('Trained StyleGAN-FineTune-v5 load successful!! 😊')

    # image generation test
    run_inference_test_after_finetuning(finetune_v5_generator, trunc_psi=1.0, epoch_no_temp=0)
    run_inference_test_after_finetuning(finetune_v5_generator, trunc_psi=0.75, epoch_no_temp=1)
    run_inference_test_after_finetuning(finetune_v5_generator, trunc_psi=0.5, epoch_no_temp=2)
    run_inference_test_after_finetuning(finetune_v5_generator, trunc_psi=0.25, epoch_no_temp=3)
    run_inference_test_after_finetuning(finetune_v5_generator, trunc_psi=0.0, epoch_no_temp=4)
