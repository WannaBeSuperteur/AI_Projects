from torchinfo import summary
from torchview import draw_graph

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/property_score_cnn/model_structure_pdf'


# Model Summary (모델 구조) 출력
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - model               (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator 또는 Discriminator
# - model_name          (str)       : 모델을 나타내는 이름
# - input_size          (tuple)     : 모델에 입력될 데이터의 입력 크기
# - print_layer_details (bool)      : 각 레이어 별 detailed info 출력 여부
# - print_frozen        (bool)      : 각 레이어가 freeze 되었는지의 상태 출력 여부

def print_summary(model, model_name, input_size, print_layer_details=False, print_frozen=False):
    print(f'\n\n==== MODEL SUMMARY : {model_name} ====\n')
    summary(model, input_size=input_size)

    if print_layer_details:
        print(model)

    if print_frozen:
        for name, param in model.named_parameters():
            print(f'layer name = {name:40s}, trainable = {param.requires_grad}')


# 모델의 구조를 PDF 로 내보내기
# Create Date : 2025.05.27
# Last Update Date : -

# Arguments:
# - model               (nn.Module) : PDF 로 구조를 내보낼 모델
# - model_name          (str)       : 모델을 나타내는 이름
# - input_size          (tuple)     : 모델에 입력될 데이터의 입력 크기
# - print_layer_details (bool)      : 각 레이어 별 detailed info 출력 여부
# - print_frozen        (bool)      : 각 레이어가 freeze 되었는지의 상태 출력 여부

def save_model_structure_pdf(model, model_name, input_size, print_layer_details=False, print_frozen=False):
    model_graph = draw_graph(model, input_size=input_size, depth=5)
    visual_graph = model_graph.visual_graph

    # Model Graph 이미지 저장
    os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)
    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/{model_name}.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)

    # Model Summary 출력
    print_summary(model, model_name, input_size, print_layer_details=print_layer_details, print_frozen=print_frozen)
