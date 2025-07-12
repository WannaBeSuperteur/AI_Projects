import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # remove for LLM inference only

import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tuning.fine_tuning_kanana import fine_tune_model as fine_tune_kanana
from fine_tuning.fine_tuning_kanana import get_stop_token_list as get_stop_token_list_kanana
from fine_tuning.inference import run_inference_kanana

from fine_tuning.utils import load_valid_final_prompts, get_answer_start_mark, get_temperature


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ANSWER_CNT = 4


# Fine-Tuning 된 LLM 로딩
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - llm_name (str) : Fine-Tuning 된 LLM 의 이름 ('kanana', 'kananai')

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 LLM

def load_fine_tuned_llm(llm_name):
    fine_tuned_llm = None

    if llm_name == 'kanana':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/ai_quiz/models/kanana_sft_final_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    elif llm_name == 'kananai':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/ai_quiz/models/kananai_sft_final_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    return fine_tuned_llm


# LLM inference (해당 LLM 이 없거나 로딩 실패 시 Fine-Tuning 학습) 실시
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - llm_name   (str) : Inference 또는 Fine-Tuning 할 LLM 의 이름 ('kanana', 'kananai')

def inference_or_fine_tune_llm(llm_name):
    models_dir = f'{PROJECT_DIR_PATH}/ai_quiz/models'

    # load valid dataset
    valid_final_input_prompts = load_valid_final_prompts()

    for final_input_prompt in valid_final_input_prompts:
        print(f'final input prompt for validation : {final_input_prompt}')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm(llm_name)
        tokenizer = AutoTokenizer.from_pretrained(f'{models_dir}/{llm_name}_sft_final_fine_tuned')
        print(f'Fine-Tuned LLM ({llm_name}) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'Fine-Tuned LLM ({llm_name}) load failed : {e}')

        if llm_name == 'kanana':
            fine_tune_kanana(instruct_version=False)

        elif llm_name == 'kananai':
            fine_tune_kanana(instruct_version=True)

        fine_tuned_llm = load_fine_tuned_llm(llm_name)
        tokenizer = AutoTokenizer.from_pretrained(f'{models_dir}/{llm_name}_sft_final_fine_tuned')

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    inference_temperature = get_temperature()
    llm_log_path = f'{PROJECT_DIR_PATH}/ai_quiz/llm_fine_tuning/logs'
    inference_log_path = f'{llm_log_path}/{llm_name}_sft_final_inference_log_{inference_temperature}.txt'
    inference_log = ''

    # run inference using Fine-Tuned LLM
    for final_input_prompt in valid_final_input_prompts:
        llm_input_print = f'\nLLM input :\n{final_input_prompt}'
        print(llm_input_print)

        # generate 4 answers for comparison
        llm_answers = []
        trial_counts = []
        output_token_cnts = []
        elapsed_times = []

        for _ in range(ANSWER_CNT):
            llm_answer, trial_count, output_token_cnt = None, None, None
            answer_start_mark = get_answer_start_mark()

            inference_start_at = time.time()

            if llm_name == 'kanana' or llm_name == 'kananai':
                stop_token_list = get_stop_token_list_kanana()
                llm_answer, trial_count, output_token_cnt = run_inference_kanana(fine_tuned_llm,
                                                                                 final_input_prompt,
                                                                                 tokenizer,
                                                                                 stop_token_list=stop_token_list,
                                                                                 answer_start_mark=answer_start_mark)

            elapsed_time = time.time() - inference_start_at

            llm_answers.append(llm_answer)
            trial_counts.append(str(trial_count))
            output_token_cnts.append(str(output_token_cnt))
            elapsed_times.append(elapsed_time)

        trial_counts_str = ','.join(trial_counts)
        output_token_cnts_str = ','.join(output_token_cnts)
        llm_answers_and_times = [f'{llm_answers[i]} (🕚 {round(elapsed_times[i], 2)} s)' for i in range(ANSWER_CNT)]
        llm_answers_and_times_str = '\n- '.join(llm_answers_and_times)

        llm_output_print = (f'Oh-LoRA answer (trials: {trial_counts_str} | output_tkn_cnt : {output_token_cnts_str}) '
                            f':\n- {llm_answers_and_times_str}')
        print(llm_output_print)

        # write inference log
        inference_log += '\n' + llm_input_print + '\n' + llm_output_print

        f = open(inference_log_path, 'w', encoding='UTF8')
        f.write(inference_log)
        f.close()


if __name__ == '__main__':

    # parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-llm_names',
                        help="name of LLMs (separated by comma)",
                        default='kananai')
    args = parser.parse_args()
    llm_names = args.llm_names
    llm_names_list = llm_names.split(',')

    for llm_name in llm_names_list:
        assert llm_name in ['kanana', 'kananai'], "LLM name must be 'kanana', 'kananai'."

        print(f'\n=== 🚀 Fine-Tune LLM {llm_name} START 🚀 ===')
        inference_or_fine_tune_llm(llm_name)
