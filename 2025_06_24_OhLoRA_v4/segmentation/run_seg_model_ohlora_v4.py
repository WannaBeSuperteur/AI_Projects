
import time
import numpy as np
import torch

from seg_model_ohlora_v4.model import SegModelForOhLoRAV4
from seg_model_ohlora_v4.train import main as main_train
from common import save_model_structure_pdf
from torchinfo import summary

SEG_IMAGE_SIZE = 224
PDF_BATCH_SIZE = 10


# Model Structure PDF 저장
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - seg_model (nn.Module) : Oh-LoRA v4 용 경량화된 Segmentation Model

def save_model_structure_pdf_files(seg_model):
    input_size = (PDF_BATCH_SIZE, 3, SEG_IMAGE_SIZE, SEG_IMAGE_SIZE)
    summary(seg_model, input_size=input_size)

    depths = [1, 2, 5]
    suffices = ['bird_eye_view', 'intermediate', 'deep_dive']

    for depth, suffix in zip(depths, suffices):
        save_model_structure_pdf(seg_model,
                                 model_name=f'seg_model_ohlora_v4_{suffix}',
                                 input_size=input_size,
                                 depth=depth)


# 경량화된 Segmentation Model 실행 시간 측정
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - seg_model (nn.Module) : Oh-LoRA v4 용 경량화된 Segmentation Model

def measure_inference_time(seg_model):
    inference_time_record = []
    random_tensor = torch.randn(1, 3, SEG_IMAGE_SIZE, SEG_IMAGE_SIZE)

    for _ in range(100):
        start_at = time.time()
        seg_model(random_tensor.cuda())
        inference_time = time.time() - start_at
        inference_time_record.append(inference_time)

    print(f'max        inference time : {max(inference_time_record)}')
    print(f'min        inference time : {min(inference_time_record)}')
    print(f'mean    of inference time : {np.mean(inference_time_record)}')
    print(f'std-dev of inference time : {np.std(inference_time_record)}')
    print(f'median  of inference time : {np.median(inference_time_record)}')


if __name__ == '__main__':
    seg_model = SegModelForOhLoRAV4()
    save_model_structure_pdf_files(seg_model)
    measure_inference_time(seg_model)

    # train Segmentation Model for Oh-LoRA v4
    main_train()
