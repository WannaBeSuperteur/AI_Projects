import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from common_values import PROMPT_PREFIX, PROMPT_SUFFIX
from sklearn.model_selection import train_test_split

import pandas as pd


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
#                                   columns: ['input_data', 'output_data', 'dest_shape_info', 'score']

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
#                                   columns: ['input_data', 'output_data', 'dest_shape_info', 'score']

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
# - llm              (LLM)       : SFT + ORPO 로 Fine-tuning 된 LLM
# - llm_prompts      (list(str)) : 해당 LLM 에 전달할 User Prompt (Prompt Engineering 을 위해 추가한 부분 제외)
# - llm_dest_outputs (list(str)) : 해당 LLM 의 목표 output 답변

# Returns:
# - llm_answers (list(str)) : 해당 LLM 의 답변
# - score       (float)     : 해당 LLM 의 성능 score

def test_orpo_llm(llm, llm_prompts, llm_dest_outputs):
    raise NotImplementedError


if __name__ == '__main__':

    # check cuda is available
    assert torch.cuda.is_available(), "CUDA MUST BE AVAILABLE"
    print(f'cuda is available with device {torch.cuda.get_device_name()}')

    orpo_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/orpo_dataset_llm.csv'
    df = pd.read_csv(orpo_dataset_path)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=2025)

    # LLM Fine-tuning
    llm = run_fine_tuning(df_train)
    save_orpo_llm(llm)

    # LLM 테스트
    llm = load_orpo_llm()
    llm_prompts = df_valid['input_data'].tolist()
    llm_dest_outputs = df_valid['output_data'].tolist()

    llm_answer, score = test_orpo_llm(llm, llm_prompts, llm_dest_outputs)

    print(f'\nLLM Score :\n{score}')
