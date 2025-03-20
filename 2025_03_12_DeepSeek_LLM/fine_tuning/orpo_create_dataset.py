from sft_fine_tuning import load_sft_llm
import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# ORPO 용 데이터셋 생성할 때, 각 LLM output 의 score 평가
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - input_data  (str) : 사용자 입력 프롬프트 + Prompt Engeineering 된 부분
# - output_data (str) : LLM 의 출력 답변

# Returns:
# - score (float) : input_data 에 대한 output_data 의 적절성 평가 score

def compute_output_score(input_data, output_data):
    raise NotImplementedError


# ORPO 용 데이터셋 (SFT 와 동일한 포맷) 의 DataFrame 에 ORPO 용 데이터 추가
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - llm           (LLM)              : SFT 로 Fine-tuning 된 LLM
# - sft_format_df (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame (from create_dataset/orpo_dataset_llm.csv)
#                                      columns: ['input_data', 'output_data', 'score']

# Returns:
# - orpo_df (Pandas DataFrame) : ORPO 용 데이터가 추가된 데이터셋의 DataFrame
#                                columns: ['input_data', 'output_data', 'score']

def add_orpo_dataset(llm, sft_format_df):
    raise NotImplementedError


if __name__ == '__main__':
    llm = load_sft_llm()
    sft_format_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/orpo_dataset_llm.csv'
    sft_format_df = pd.read_csv(sft_format_dataset_path)

    # 이미 ORPO 용 데이터가 추가된 경우 오류 반환
    assert min(sft_format_df['score']) < 1.0, 'ORPO DATA ALREADY ADDED'

    orpo_df = add_orpo_dataset(llm, sft_format_df)
    orpo_df.to_csv(f'{PROJECT_DIR_PATH}/create_dataset/orpo_dataset_llm.csv')