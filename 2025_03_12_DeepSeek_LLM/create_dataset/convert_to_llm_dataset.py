import pandas as pd
import os

# *.csv 데이터셋을 LLM 이 직접 학습 가능한 학습 데이터셋으로 변환
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - csv_path     (str) : *.csv 데이터셋 파일 경로
# - train_option (str) : 학습 방법, 'sft' 또는 'orpo'

# Returns:
# - LLM 이 직접 학습 가능한 데이터셋 파일을 저장

def convert_to_llm_data(csv_path, train_option):
    assert train_option in ['sft', 'orpo']

    df = pd.read_csv(csv_path)
    df = df[['input_data', 'output_data']]

    # ORPO 의 경우 SFT 의 학습 데이터셋과 동일한 포맷의 input, output 은 score = 1.0 으로 처리
    if train_option == 'orpo':
        df['score'] = 1.0

    # shuffle rows
    df = df.sample(frac=1)

    save_path = csv_path.replace('_dataset.csv', '_dataset_llm.csv')
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))

    # Supervised Fine-Tuning 데이터셋 생성
    convert_to_llm_data(f'{abs_path}/sft_dataset.csv', 'sft')

    # ORPO 데이터셋 생성
    convert_to_llm_data(f'{abs_path}/orpo_dataset.csv', 'orpo')
