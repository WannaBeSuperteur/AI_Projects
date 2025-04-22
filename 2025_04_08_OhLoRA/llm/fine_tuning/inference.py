import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


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


# Fine Tuning 된 LLM (gemma-2 2b) 을 이용한 inference 실시
# Create Date : 2025.04.22
# Last Update Date : 2025.04.22
# - 변경된 LLM input data format 반영

# Arguments:
# - fine_tuned_llm (LLM)           : Fine-Tuning 된 LLM
# - user_prompt    (str)           : LLM 에 입력할 사용자 프롬프트
# - tokenizer      (AutoTokenizer) : LLM 의 Tokenizer

# Returns:
# - llm_answer (str) : LLM 답변 중 user prompt 를 제외한 부분

def run_inference(fine_tuned_llm, user_prompt, tokenizer):
    user_prompt_ = user_prompt + ' (answer start)'
    inputs = tokenizer(user_prompt_, return_tensors='pt').to(fine_tuned_llm.device)

    outputs = fine_tuned_llm.generate(**inputs, max_length=80, do_sample=True)
    llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    llm_answer = llm_answer[len(user_prompt_):]

    return llm_answer
