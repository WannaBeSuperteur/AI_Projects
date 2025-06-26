
from torchview import draw_graph
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MODEL_STRUCTURE_PDF_DIR_PATH = f'{PROJECT_DIR_PATH}/segmentation/model_structure_pdf'
os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)


# 기존 모델의 구조를 PDF 로 내보내기
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model      (nn.Module) : 기존 Pre-train 된 StyleGAN 모델의 Generator 또는 Discriminator
# - model_name (str)       : 모델을 나타내는 이름
# - input_size (tuple)     : 모델에 입력될 데이터의 입력 크기
# - depth      (int)       : 모델 PDF에 표시할 모델 구조의 depth (값이 작을수록 bird-eye view, 클수록 deep dive)

def save_model_structure_pdf(model, model_name, input_size, depth):
    model_graph = draw_graph(model, input_size=input_size, depth=depth)
    visual_graph = model_graph.visual_graph

    # Model Graph 이미지 저장
    os.makedirs(MODEL_STRUCTURE_PDF_DIR_PATH, exist_ok=True)
    dest_name = f'{MODEL_STRUCTURE_PDF_DIR_PATH}/{model_name}.pdf'
    visual_graph.render(format='pdf', outfile=dest_name)
