import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tuning.fine_tuning_polyglot import fine_tune_model
from fine_tuning.inference import run_inference, load_valid_user_prompts


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Fine Tuning 된 LLM (Polyglot-Ko 1.3B) 로딩
# Create Date : 2025.04.22
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 Polyglot-Ko 1.3B LLM

def load_fine_tuned_llm():
    fine_tuned_llm = AutoModelForCausalLM.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/polyglot_fine_tuned',
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).cuda()
    return fine_tuned_llm


if __name__ == '__main__':

    # load valid dataset
    valid_user_prompts = load_valid_user_prompts()

    for user_prompt in valid_user_prompts:
        print(f'user prompt for validation : {user_prompt}')

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/polyglot_fine_tuned')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm()
        print('Fine-Tuned LLM of Polyglot-Ko 1.3B - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'Fine-Tuned LLM (Polyglot-Ko 1.3B) load failed : {e}')

        fine_tune_model()
        fine_tuned_llm = load_fine_tuned_llm()

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    # run inference using Fine-Tuned LLM
    for user_prompt in valid_user_prompts:
        print(f'\nuser prompt :\n{user_prompt}')

        # generate 4 answers for comparison
        llm_answer_0, trial_count_0, output_token_cnt_0 = run_inference(fine_tuned_llm,
                                                                        user_prompt,
                                                                        tokenizer,
                                                                        answer_start_mark=' (답변 시작)',
                                                                        remove_token_type_ids=True)

        llm_answer_1, trial_count_1, output_token_cnt_1 = run_inference(fine_tuned_llm,
                                                                        user_prompt,
                                                                        tokenizer,
                                                                        answer_start_mark=' (답변 시작)',
                                                                        remove_token_type_ids=True)

        llm_answer_2, trial_count_2, output_token_cnt_2 = run_inference(fine_tuned_llm,
                                                                        user_prompt,
                                                                        tokenizer,
                                                                        answer_start_mark=' (답변 시작)',
                                                                        remove_token_type_ids=True)

        llm_answer_3, trial_count_3, output_token_cnt_3 = run_inference(fine_tuned_llm,
                                                                        user_prompt,
                                                                        tokenizer,
                                                                        answer_start_mark=' (답변 시작)',
                                                                        remove_token_type_ids=True)

        print(f'Oh-LoRA answer (trials: {trial_count_0},{trial_count_1},{trial_count_2},{trial_count_3} | '
              f'output_tkn_cnt : {output_token_cnt_0},{output_token_cnt_1},{output_token_cnt_2},{output_token_cnt_3}) '
              f':\n- {llm_answer_0}\n- {llm_answer_1}\n- {llm_answer_2}\n- {llm_answer_3}')
