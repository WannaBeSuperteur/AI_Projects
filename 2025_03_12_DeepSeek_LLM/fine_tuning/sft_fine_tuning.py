import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# for prompt engineering
PROMPT_PREFIX = "Represent below as a Python list.\n\n"
PROMPT_SUFFIX = """in the following format.

At this time, each node is represented in the format of Python list "[node No.,
X position (px), Y position (px), shape (rectangle, round rectangle or circle),
width (px), height (px), connection line shape (solid or dashed), background color,
connection line color, list of node No. s of other nodes pointed to by the connection line]".

At this time, the color is represented in the format of tuple (R, G, B), between 0 and 255, and
X position range is 0-1000 and Y position range is 0-600.

It is important to draw a representation of high readability."""

TEST_PROMPT = ("CNN with 128 x 128 input size, a 3 x 3 convolutional layer and a 2 x 2 pooling layer, " +
               "then 2 3 x 3 convolutional layers and a 2 x 2 pooling layer, 2 3 x 3 convolutional layers " +
               "and a 2 x 2 pooling layer, a 3 x 3 conv layer and a 2 x 2 pooling layer, " +
               "then and 1024 nodes in hiddens, and 1 output size")


# SFT 실시
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame
#                                   columns: ['input_data', 'output_data']

# Returns:
# - llm (LLM) : SFT 로 Fine-tuning 된 LLM

def run_fine_tuning(dataset_df):
    raise NotImplementedError


# SFT 테스트를 위한 모델 로딩
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - llm (LLM) : SFT 로 Fine-tuning 된 LLM

def load_sft_llm():
    raise NotImplementedError


# SFT 로 Fine-Tuning 된 LLM 저장
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - llm (LLM) : SFT 로 Fine-tuning 된 LLM

# Returns:
# - 해당 LLM 을 파일로 저장

def save_sft_llm(llm):
    raise NotImplementedError


# SFT 로 Fine-Tuning 된 LLM 을 테스트
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - llm (LLM) : SFT 로 Fine-tuning 된 LLM
# - llm_prompt (str) : 해당 LLM 에 전달할 User Prompt (Prompt Engineering 을 위해 추가한 부분 제외)

# Returns:
# - llm_answer (str) : 해당 LLM 의 답변

def test_sft_llm(llm, llm_prompt):
    raise NotImplementedError


if __name__ == '__main__':
    sft_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/sft_dataset_llm.csv'
    df = pd.read_csv(sft_dataset_path)

    # LLM Fine-tuning
    llm = run_fine_tuning(df)
    save_sft_llm(llm)

    # LLM 테스트
    llm = load_sft_llm()
    llm_prompt = TEST_PROMPT
    llm_answer = test_sft_llm(llm, llm_prompt)

    print(f'LLM Prompt:\n{llm_prompt}')
    print(f'\nLLM Answer:\n{llm_answer}')