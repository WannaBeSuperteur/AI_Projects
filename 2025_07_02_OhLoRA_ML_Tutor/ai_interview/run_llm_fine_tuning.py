import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # remove for LLM inference only

import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fine_tuning.fine_tuning_kanana import fine_tune_model as fine_tune_kanana
from llm_fine_tuning.fine_tuning_kanana import get_stop_token_list as get_stop_token_list_kanana
from llm_fine_tuning.fine_tuning_midm import fine_tune_model as fine_tune_midm
from llm_fine_tuning.fine_tuning_midm import get_stop_token_list as get_stop_token_list_midm
from llm_fine_tuning.inference import run_inference_kanana, run_inference_midm

from llm_fine_tuning.utils import load_valid_final_prompts, get_answer_start_mark, get_temperature


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ANSWER_CNT = 4


# Fine-Tuning 된 LLM 로딩
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - llm_name      (str) : Fine-Tuning 된 LLM 의 이름 ('kanana', 'kananai' or 'midm')
# - kanana_epochs (int) : Kanana-1.5 2.1B instruct 모델 학습 시의 epochs
# - midm_epochs   (int) : 믿음:2.0 모델 학습 시의 epochs

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 LLM

def load_fine_tuned_llm(llm_name, kanana_epochs, midm_epochs):
    fine_tuned_llm = None

    if llm_name == 'kanana':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/ai_interview/models/kanana_sft_final_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    elif llm_name == 'kananai':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/ai_interview/models/kananai_sft_final_fine_tuned_{kanana_epochs}epochs',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    elif llm_name == 'midm':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/ai_interview/models/midm_sft_final_fine_tuned_{midm_epochs}epochs',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    return fine_tuned_llm


# LLM inference (해당 LLM 이 없거나 로딩 실패 시 Fine-Tuning 학습) 실시
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - llm_name (str) : Inference 또는 Fine-Tuning 할 LLM 의 이름 ('kanana', 'kananai' or 'midm')
# - epochs   (int) : 학습 epoch 횟수

def inference_or_fine_tune_llm(llm_name, epochs):
    models_dir = f'{PROJECT_DIR_PATH}/ai_interview/models'

    # load valid dataset
    valid_final_input_prompts = load_valid_final_prompts()

    for final_input_prompt in valid_final_input_prompts:
        print(f'final input prompt for validation :\n{final_input_prompt}\n')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm(llm_name, kanana_epochs=epochs, midm_epochs=epochs)
        tokenizer = AutoTokenizer.from_pretrained(f'{models_dir}/{llm_name}_sft_final_fine_tuned_{epochs}epochs')
        print(f'Fine-Tuned LLM ({llm_name}) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'Fine-Tuned LLM ({llm_name}) load failed : {e}')

        if llm_name == 'kanana':
            fine_tune_kanana(instruct_version=False, epochs=epochs)

        elif llm_name == 'kananai':
            fine_tune_kanana(instruct_version=True, epochs=epochs)

        elif llm_name == 'midm':
            fine_tune_midm(epochs=epochs)

        fine_tuned_llm = load_fine_tuned_llm(llm_name, kanana_epochs=epochs, midm_epochs=epochs)
        tokenizer = AutoTokenizer.from_pretrained(f'{models_dir}/{llm_name}_sft_final_fine_tuned_{epochs}epochs')

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    inference_temperature = get_temperature()
    llm_log_path = f'{PROJECT_DIR_PATH}/ai_interview/llm_fine_tuning/logs'
    inference_log_path = f'{llm_log_path}/{llm_name}_sft_final_inference_log_{inference_temperature}_{epochs}epochs.txt'
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

            elif llm_name == 'midm':
                stop_token_list = get_stop_token_list_midm()
                llm_answer, trial_count, output_token_cnt = run_inference_midm(fine_tuned_llm,
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
    parser.add_argument('-epochs',
                        help="name of epochs for train (separated by comma)",
                        default='5')

    args = parser.parse_args()
    print(f'arguments : {args}')

    llm_names, epochs = args.llm_names, args.epochs
    llm_names_list, epochs_list = llm_names.split(','), epochs.split(',')

    for llm_name, epochs in zip(llm_names_list, epochs_list):
        assert llm_name in ['kanana', 'kananai', 'midm'], "LLM name must be 'kanana', 'kananai' or 'midm'."

        print(f'\n=== 🚀 Fine-Tune LLM {llm_name} START 🚀 ===')
        inference_or_fine_tune_llm(llm_name, int(epochs))
