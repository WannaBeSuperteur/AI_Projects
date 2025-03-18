import pandas as pd
import common
import os

# LLM 의 Supervised Fine-tuning (SFT) 학습 데이터셋 생성
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - sft_dataset.csv 파일에 SFT dataset 저장

def generate_sft_dataset():
    dl_dataset = common.generate_dl_model_dataset(dataset_size=280)
    dataset = pd.concat([dl_dataset], axis=0)

    abs_path = os.path.abspath(os.path.dirname(__file__))
    dataset.to_csv(f'{abs_path}/sft_dataset.csv')


if __name__ == '__main__':
    generate_sft_dataset()
