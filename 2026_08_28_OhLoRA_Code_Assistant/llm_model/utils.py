
import os
from datetime import datetime

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


LLM_ORIGINAL_PATHS = ['kakaocorp/kanana-2-3b-instruct',
                      'kakaocorp/kanana-1.5-2.1b-instruct-2505',
                      'K-intelligence/Midm-2.0-Mini-Instruct',
                      'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B']

TARGET_MODULES_DICT = {
    'default': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

DATASET_PATH = f'{PROJECT_DIR_PATH}/llm/train_data.csv'


STOP_TOKEN_LIST = {
    'default': [109659, 104449, 99458, 64356, 8, 220]  # (답변 종료)"
}
ANSWER_START_MARK = ' (답변 시작)'
ANSWER_END_MARK = ' (답변 종료)'


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