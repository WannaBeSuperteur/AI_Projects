import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common_values import PROMPT_PREFIX, PROMPT_SUFFIX

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

TEST_PROMPT = ("CNN with 128 x 128 input size, a 3 x 3 convolutional layer and a 2 x 2 pooling layer, " +
               "then 2 3 x 3 convolutional layers and a 2 x 2 pooling layer, 2 3 x 3 convolutional layers " +
               "and a 2 x 2 pooling layer, a 3 x 3 conv layer and a 2 x 2 pooling layer, " +
               "then and 1024 nodes in hiddens, and 1 output size")


# ORPO Fine-Tuning 을 위해 Pandas DataFrame 을 ORPO 형식 {"prompt": [...], "chosen": [...], "rejected": [...]} 으로 변환
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame
#                                   columns: ['input_data', 'output_data', 'score']

# Returns:
# - orpo_format_dataset (dict(list)) : ORPO 로 직접 학습 가능한 데이터 형식으로 변환된 데이터셋
#                                      형식: {"prompt": [...], "chosen": [...], "rejected": [...]}

def convert_df_to_orpo_format(dataset_df):
    raise NotImplementedError


# ORPO Fine-Tuning 실시
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame
#                                   columns: ['input_data', 'output_data', 'score']

# Returns:
# - llm (LLM) : SFT 로 Fine-tuning 된 LLM

def run_fine_tuning(dataset_df):
    orpo_format_dataset = convert_df_to_orpo_format(dataset_df)

    raise NotImplementedError


# ORPO 테스트를 위한 모델 로딩
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - llm (LLM) : SFT + ORPO 로 Fine-tuning 된 LLM

def load_orpo_llm():
    raise NotImplementedError


# SFT + ORPO 로 Fine-Tuning 된 LLM 저장
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - llm (LLM) : SFT + ORPO 로 Fine-tuning 된 LLM

# Returns:
# - 해당 LLM 을 파일로 저장

def save_orpo_llm(llm):
    raise NotImplementedError


# SFT + ORPO 로 Fine-Tuning 된 LLM 을 테스트
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - llm (LLM) : SFT + ORPO 로 Fine-tuning 된 LLM
# - llm_prompt (str) : 해당 LLM 에 전달할 User Prompt (Prompt Engineering 을 위해 추가한 부분 제외)

# Returns:
# - llm_answer (str) : 해당 LLM 의 답변

def test_orpo_llm(llm, llm_prompt):
    raise NotImplementedError


if __name__ == '__main__':
    orpo_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/orpo_dataset_llm.csv'
    df = pd.read_csv(orpo_dataset_path)

    # LLM Fine-tuning
    llm = run_fine_tuning(df)
    save_orpo_llm(llm)

    # LLM 테스트
    llm = load_orpo_llm()
    llm_prompt = TEST_PROMPT
    llm_answer = test_orpo_llm(llm, llm_prompt)

    print(f'LLM Prompt:\n{llm_prompt}')
    print(f'\nLLM Answer:\n{llm_answer}')