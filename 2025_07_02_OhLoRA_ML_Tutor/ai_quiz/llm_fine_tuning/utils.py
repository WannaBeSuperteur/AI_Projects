import pandas as pd
import os
from datetime import datetime

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


def get_answer_start_mark():
    return ' (답변 시작)'


def get_answer_end_mark():
    return ' (답변 종료)'


def get_temperature():
    return 0.6


def preview_dataset(dataset, tokenizer, print_encoded_tokens=False):
    print('\n=== DATASET PREVIEW ===')
    print(f"dataset size : [train: {len(dataset['train']['text'])}, valid: {len(dataset['valid']['text'])}]")

    for i in range(10):
        print(f"\ntrain data {i} : {dataset['train']['text'][i]}")
        if print_encoded_tokens:
            print(f"train data {i} tokenized : {tokenizer.encode(dataset['train']['text'][i])}")

        print(f"valid data {i} : {dataset['valid']['text'][i].split('###')[0]}")
        if print_encoded_tokens:
            print(f"valid data {i} tokenized : {tokenizer.encode(dataset['valid']['text'][i].split('###')[0])}")

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
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - valid_final_prompts (list(str)) : Valid Dataset 로부터 가져온 final LLM input prompt 의 리스트

def load_valid_final_prompts():
    dataset_csv_path = 'ai_qna/fine_tuning_dataset/SFT_final.csv'
    dataset_csv_path = f'{PROJECT_DIR_PATH}/{dataset_csv_path}'
    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df_valid = dataset_df[dataset_df['data_type'].str.startswith('valid')]

    valid_final_prompts = dataset_df_valid['input_data'].tolist()
    return valid_final_prompts
