import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tuning.fine_tuning import fine_tune_model
from fine_tuning.inference import run_inference, load_valid_user_prompts


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


if __name__ == '__main__':

    # load valid dataset
    valid_user_prompts = load_valid_user_prompts()

    for user_prompt in valid_user_prompts:
        print(f'user prompt for validation : {user_prompt}')

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/original')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm()

    except Exception as e:
        print(f'Fine-Tuned LLM load failed : {e}')

        fine_tune_model()
        fine_tuned_llm = load_fine_tuned_llm()

    # run inference using Fine-Tuned LLM
    for user_prompt in valid_user_prompts:
        print(f'\nuser prompt :\n{user_prompt}')

        # generate 2 answers for comparison
        llm_answer_0, trial_count_0 = run_inference(fine_tuned_llm, user_prompt, tokenizer)
        llm_answer_1, trial_count_1 = run_inference(fine_tuned_llm, user_prompt, tokenizer)

        print(f'Oh-LoRA answer (trials: {trial_count_0},{trial_count_1}) :\n- {llm_answer_0}\n- {llm_answer_1}')
