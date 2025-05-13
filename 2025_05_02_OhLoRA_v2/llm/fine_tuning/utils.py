import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


def get_instruction():
    return '당신은 AI 여성 챗봇입니다. 사용자의 대화에 답하세요.'


def preview_dataset(dataset):
    for i in range(10):
        print(f"train data {i} : {dataset['train']['text'][i]}")
        print(f"valid data {i} : {dataset['valid']['text'][i].split('###')[0]}")


# Valid Dataset 에 있는 user prompt 가져오기 (테스트 데이터셋 대용)
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - 업데이트된 학습 데이터셋 (OhLoRA_fine_tuning_v2.csv) 반영, 총 4 개의 LLM 학습 로직 적용
# - LLM 4개 학습 로직에 맞게, 함수명 및 변수명 수정

# Arguments:
# - dataset_csv_path (str) : Valid dataset csv 파일 경로
# - output_col       (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - valid_final_prompts (list(str)) : Valid Dataset 로부터 가져온 final LLM input prompt 의 리스트

def load_valid_final_prompts(dataset_csv_path, output_col):
    dataset_csv_path = f'{PROJECT_DIR_PATH}/{dataset_csv_path}'
    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df_valid = dataset_df[dataset_df['data_type'] == 'valid']

    if output_col == 'summary':
        valid_final_prompts = (dataset_df_valid['input_data'] + ' / ' + dataset_df_valid['output_message']).tolist()
    else:
        valid_final_prompts = dataset_df_valid['input_data'].tolist()

    return valid_final_prompts


# Modified Implementation from https://github.com/quantumaikr/KoreanLM/blob/main/finetune-lora.py

def koreanlm_tokenize(prompt, tokenizer, return_tensors):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=96,
        padding=False,
        return_tensors=return_tensors,
    )

    result["labels"] = result["input_ids"].copy()
    return result
