import pandas as pd
import os
from datetime import datetime

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


def get_instruction():
    return '당신은 AI 여성 챗봇입니다. 사용자의 대화에 답하세요.'


def preview_dataset(dataset):
    print('\n=== DATASET PREVIEW ===')
    for i in range(10):
        print(f"train data {i} : {dataset['train']['text'][i]}")
        print(f"valid data {i} : {dataset['valid']['text'][i].split('###')[0]}")
    print('')


def add_train_log(state, train_log_dict):
    last_log = state.log_history[-1]
    batch_cnt_per_epoch = state.max_steps // state.num_train_epochs
    is_first_log_of_epoch = (0 < last_log['step'] % batch_cnt_per_epoch <= state.logging_steps)

    if is_first_log_of_epoch:
        train_log_dict['time'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        train_log_dict['time'].append('-')

    train_log_dict['epoch'].append(round(last_log['epoch'], 2))
    train_log_dict['loss'].append(round(last_log['loss'], 4))
    train_log_dict['grad_norm'].append(round(last_log['grad_norm'], 4))
    train_log_dict['learning_rate'].append(round(last_log['learning_rate'], 6))
    train_log_dict['mean_token_accuracy'].append(round(last_log['mean_token_accuracy'], 4))


def add_inference_log(inference_result, inference_log_dict):
    inference_log_dict['epoch'].append(int(inference_result['epoch']))
    inference_log_dict['elapsed_time (s)'].append(round(inference_result['elapsed_time'], 2))
    inference_log_dict['prompt'].append(inference_result['prompt'])
    inference_log_dict['llm_answer'].append(inference_result['llm_answer'])
    inference_log_dict['trial_cnt'].append(inference_result['trial_cnt'])
    inference_log_dict['output_tkn_cnt'].append(inference_result['output_tkn_cnt'])


# Valid Dataset 에 있는 user prompt 가져오기 (테스트 데이터셋 대용)
# Create Date : 2025.05.12
# Last Update Date : 2025.05.14
# - Oh-LoRA Fine-Tuning 학습 데이터셋 v2.1 (OhLoRA_fine_tuning_v2_1.csv) 반영, csv path 인수 제거

# Arguments:
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - valid_final_prompts (list(str)) : Valid Dataset 로부터 가져온 final LLM input prompt 의 리스트

def load_valid_final_prompts(output_col):
    if output_col in ['output_message', 'summary']:
        dataset_csv_path = 'llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2_1.csv'
    else:
        dataset_csv_path = 'llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv'

    dataset_csv_path = f'{PROJECT_DIR_PATH}/{dataset_csv_path}'
    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df_valid = dataset_df[dataset_df['data_type'] == 'valid']

    if output_col == 'summary':
        valid_final_prompts = (dataset_df_valid['input_data'] + ' / ' + dataset_df_valid['output_message']).tolist()
    elif output_col == 'eyes_mouth_pose':
        valid_final_prompts = dataset_df_valid['output_message'].tolist()
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
