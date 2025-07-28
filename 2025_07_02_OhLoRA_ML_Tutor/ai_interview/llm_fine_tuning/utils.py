
import pandas as pd
import numpy as np

import os
from datetime import datetime


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


def get_answer_start_mark():
    return ' (발화 시작)'


def get_answer_end_mark():
    return ' (발화 종료)'


def get_temperature():
    return 0.6


def preview_dataset(dataset, tokenizer, print_encoded_tokens=True):
    print('\n=== DATASET PREVIEW ===')
    print(f"dataset size : [train: {len(dataset['train']['text'])}, valid: {len(dataset['valid']['text'])}]")

    for i in range(10):
        print(f"\ntrain data {i} :\n{dataset['train']['text'][i]}")
        if print_encoded_tokens:
            print(f"train data {i} tokenized : {tokenizer.encode(dataset['train']['text'][i])}")

        print(f"\nvalid data {i} :\n{dataset['valid']['text'][i].split('###')[0]}")
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


# quiz, keyword, good_answer 의 빈칸이 채워진 DataFrame 반환
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - csv_path  (str) : DataFrame 이 있는 csv 파일의 path
# - data_type (str) : 필터링할 data 유형 ('train' 또는 'valid/test', null (None) 가능)

# Returns
# - filled_dataset_df (Pandas DataFrame) : 빈칸이 채워진 DataFrame

def convert_into_filled_df(csv_path, data_type=None):
    dataset_df = pd.read_csv(csv_path)

    if data_type is not None:
        dataset_df = dataset_df[dataset_df['data_type'] == data_type]

    # get current columns
    data_type_list = dataset_df['data_type'].tolist()
    input_list = dataset_df['input_data_wo_rag_augment'].tolist()
    output_list = dataset_df['output_data'].tolist()

    # fill quiz and good answer list
    final_input_list = []
    final_output_list = []
    final_data_type_list = []

    for data_type_, input_, output_ in zip(data_type_list, input_list, output_list):
        is_input_valid = not (pd.isna(input_) or 'nan' in str(input_).lower() or input_ == '')
        is_output_valid = not (pd.isna(output_) or 'nan' in str(output_).lower() or output_ == '')
        is_valid = (is_input_valid and is_output_valid) if data_type_ == 'train' else is_input_valid

        if is_valid:
            final_input_list.append(input_)
            final_output_list.append(output_)
            final_data_type_list.append(data_type_)

    # create final filled DataFrame
    final_dataset_dict = {
        'data_type': final_data_type_list,
        'input_data': final_input_list,
        'output_data': final_output_list
    }

    filled_dataset_df = pd.DataFrame(final_dataset_dict)
    return filled_dataset_df


# Valid Dataset 에 있는 user prompt 가져오기 (테스트 데이터셋 대용)
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - valid_final_prompts (list(str)) : Valid Dataset 로부터 가져온 final LLM input prompt 의 리스트

def load_valid_final_prompts():
    dataset_csv_path = 'ai_interview/dataset/all_train_and_test_data.csv'
    dataset_csv_path = f'{PROJECT_DIR_PATH}/{dataset_csv_path}'
    dataset_df_valid = convert_into_filled_df(dataset_csv_path, data_type='valid/test')

    valid_final_prompts = dataset_df_valid['input_data'].tolist()
    return valid_final_prompts
