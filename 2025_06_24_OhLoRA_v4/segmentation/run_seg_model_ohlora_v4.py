
from seg_model_ohlora_v4.model import SegModelForOhLoRAV4
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


if __name__ == '__main__':
    seg_model = SegModelForOhLoRAV4()
    save_model_structure_pdf_files(seg_model)

