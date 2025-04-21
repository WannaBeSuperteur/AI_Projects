import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fine_tuning.fine_tuning import fine_tune_model
from fine_tuning.inference import run_inference


# Fine Tuning 된 LLM (gemma-2 2b) 로딩
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 LLM

def load_fine_tuned_llm():
    raise NotImplementedError


# Valid Dataset 에 있는 user prompt 가져오기 (테스트 데이터셋 대용)
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - valid_user_prompts (list(str)) : Valid Dataset 에 있는 user prompt 의 리스트

def load_valid_user_prompts():
    raise NotImplementedError


if __name__ == '__main__':

    # run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm()

    except Exception as e:
        print(f'Fine-Tuned LLM load failed : {e}')

        fine_tune_model()
        fine_tuned_llm = load_fine_tuned_llm()

    # run inference using Fine-Tuned LLM
    valid_user_prompts = load_valid_user_prompts()

    for user_prompt in valid_user_prompts:
        print(f'user prompt :\n{user_prompt}')
        llm_answer = run_inference(fine_tuned_llm, user_prompt)
        print(f'Oh-LoRA answer :\n{llm_answer}')
