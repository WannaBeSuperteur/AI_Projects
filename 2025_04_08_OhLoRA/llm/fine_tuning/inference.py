# for LLM answer generation config :
# - https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig


import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# Valid Dataset 에 있는 user prompt 가져오기 (테스트 데이터셋 대용)
# Create Date : 2025.04.21
# Last Update Date : 2025.04.22
# - 학습 데이터 및 그 csv 파일 경로 수정에 따른 경로 업데이트

# Arguments:
# - 없음

# Returns:
# - valid_user_prompts (list(str)) : Valid Dataset 에 있는 user prompt 의 리스트

def load_valid_user_prompts():
    dataset_csv_path = f'{PROJECT_DIR_PATH}/llm/OhLoRA_fine_tuning_25042210.csv'
    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df_valid = dataset_df[dataset_df['data_type'] == 'valid']

    valid_user_prompts = dataset_df_valid['input_data'].tolist()
    return valid_user_prompts


# Fine Tuning 된 LLM (gemma-2 2b) 을 이용한 inference 실시
# Create Date : 2025.04.22
# Last Update Date : 2025.04.22
# - 변경된 LLM input data format 반영
# - empty answer 가 생기지 않을 때까지 반복 시도

# Arguments:
# - fine_tuned_llm (LLM)           : Fine-Tuning 된 LLM
# - user_prompt    (str)           : LLM 에 입력할 사용자 프롬프트
# - tokenizer      (AutoTokenizer) : LLM 의 Tokenizer
# - max_trials     (int)           : LLM 이 empty answer 가 아닌 답변을 출력하도록 하는 최대 시도 횟수

# Returns:
# - llm_answer  (str) : LLM 답변 중 user prompt 를 제외한 부분
# - trial_count (int) : LLM 이 empty answer 가 아닌 답변을 출력하기까지의 시도 횟수

def run_inference(fine_tuned_llm, user_prompt, tokenizer, max_trials=30):
    user_prompt_ = user_prompt + ' (answer start)'
    inputs = tokenizer(user_prompt_, return_tensors='pt').to(fine_tuned_llm.device)

    llm_answer = ''
    trial_count = 0

    while trial_count < max_trials:
        outputs = fine_tuned_llm.generate(**inputs, max_length=80, do_sample=True, temperature=1.2)
        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(user_prompt_):]
        trial_count += 1

        if (llm_answer.startswith('[') and llm_answer.endswith(']')) or llm_answer.replace('\n', '') != '':
            break

    # remove new-lines
    llm_answer = llm_answer.replace('\n', '')

    return llm_answer, trial_count
