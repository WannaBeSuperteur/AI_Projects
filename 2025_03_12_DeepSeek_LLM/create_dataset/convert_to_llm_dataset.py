import pandas as pd

# *.csv 데이터셋을 LLM 이 직접 학습 가능한 학습 데이터셋으로 변환
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - csv_path     (str) : *.csv 데이터셋 파일 경로
# - train_option (str) : 학습 방법, 'sft' 또는 'orpo'

# Returns:
# - LLM 이 직접 학습 가능한 데이터셋 파일을 저장

def convert_to_llm_data(csv_path, train_option):
    raise NotImplementedError


if __name__ == '__main__':

    # Supervised Fine-Tuning 데이터셋 생성
    convert_to_llm_data('sft_dataset.py', 'sft')

    # ORPO 데이터셋 생성
    convert_to_llm_data('orpo_dataset.csv', 'orpo')
