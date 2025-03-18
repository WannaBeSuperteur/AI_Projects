import pandas as pd
import common
import os

# LLM 의 ORPO 학습 데이터셋 생성
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - orpo_dataset.csv 파일에 ORPO dataset 저장

def generate_orpo_dataset():
    dl_dataset = common.generate_dl_model_dataset(dataset_size=80)
    flow_chart_dataset = common.generate_flow_chart_dataset(dataset_size=120)
    dataset = pd.concat([dl_dataset, flow_chart_dataset], axis=0)

    abs_path = os.path.abspath(os.path.dirname(__file__))
    dataset.to_csv(f'{abs_path}/orpo_dataset.csv')


if __name__ == '__main__':
    generate_orpo_dataset()
