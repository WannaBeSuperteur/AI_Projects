import pandas as pd
import common

# LLM 의 Supervised Fine-tuning (SFT) 학습 데이터셋 생성
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - sft_dataset.csv 파일에 SFT dataset 저장

def generate_sft_dataset():
    common.generate_DL_model_dataset()


if __name__ == '__main__':
    generate_sft_dataset()
