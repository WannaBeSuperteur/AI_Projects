import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd
from transformers import AutoModelForCausalLM

from fine_tuning.fine_tuning import fine_tune_model
from fine_tuning.inference import run_inference


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Fine Tuning 된 LLM (gemma-2 2b) 로딩
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 LLM

def load_fine_tuned_llm():
    fine_tuned_llm = AutoModelForCausalLM.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/fine_tuned',
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).cuda()
    return fine_tuned_llm


# Valid Dataset 에 있는 user prompt 가져오기 (테스트 데이터셋 대용)
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - valid_user_prompts (list(str)) : Valid Dataset 에 있는 user prompt 의 리스트

def load_valid_user_prompts():
    dataset_csv_path = f'{PROJECT_DIR_PATH}/llm/OhLoRA_fine_tuning.csv'
    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df_valid = dataset_df[dataset_df['data_type'] == 'valid']

    valid_user_prompts = dataset_df_valid['input_data'].tolist()
    return valid_user_prompts


if __name__ == '__main__':

    # load valid dataset
    valid_user_prompts = load_valid_user_prompts()

    for user_prompt in valid_user_prompts:
        print(f'user prompt for validation : {user_prompt}')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm()

    except Exception as e:
        print(f'Fine-Tuned LLM load failed : {e}')

        fine_tune_model()
        fine_tuned_llm = load_fine_tuned_llm()

    # run inference using Fine-Tuned LLM
    for user_prompt in valid_user_prompts:
        print(f'user prompt :\n{user_prompt}')
        llm_answer = run_inference(fine_tuned_llm, user_prompt)
        print(f'Oh-LoRA answer :\n{llm_answer}')
