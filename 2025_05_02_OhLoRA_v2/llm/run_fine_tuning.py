import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tuning.fine_tuning_koreanlm import fine_tune_model as fine_tune_koreanlm, Prompter
from fine_tuning.fine_tuning_polyglot import fine_tune_model as fine_tune_polyglot
from fine_tuning.inference import run_inference, run_inference_koreanlm
from fine_tuning.utils import load_valid_final_prompts


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Fine-Tuning 된 LLM 로딩
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - 업데이트된 학습 데이터셋 (OhLoRA_fine_tuning_v2.csv) 반영, 총 4 개의 LLM 학습 로직 적용

# Arguments:
# - llm_name   (str) : Fine-Tuning 된 LLM 의 이름 ('polyglot' or 'koreanlm')
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 LLM

def load_fine_tuned_llm(llm_name, output_col):
    fine_tuned_llm = None

    if llm_name == 'polyglot':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/llm/models/polyglot_{output_col}_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    elif llm_name == 'koreanlm':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/llm/models/koreanlm_{output_col}_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.float16).cuda()

    return fine_tuned_llm


if __name__ == '__main__':

    # parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-llm_name', help="name of LLM ('polyglot' or 'koreanlm')", default='polyglot')
    parser.add_argument('-output_col', help="output column name from dataset csv", default='output_message')
    args = parser.parse_args()

    llm_name = args.llm_name
    output_col = args.output_col

    assert llm_name in ['polyglot', 'koreanlm'], "LLM name must be 'polyglot' or 'koreanlm'."

    # load valid dataset
    valid_final_input_prompts = load_valid_final_prompts(
        dataset_csv_path='llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv',
        output_col=output_col)

    for final_input_prompt in valid_final_input_prompts:
        print(f'final input prompt for validation : {final_input_prompt}')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm(llm_name, output_col)
        tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/{llm_name}_{output_col}_fine_tuned')
        print(f'Fine-Tuned LLM ({llm_name}) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'Fine-Tuned LLM ({llm_name}) load failed : {e}')

        if llm_name == 'koreanlm':
            fine_tune_koreanlm(output_col=output_col)

        elif llm_name == 'polyglot':
            fine_tune_polyglot(output_col=output_col)

        fine_tuned_llm = load_fine_tuned_llm(llm_name, output_col)
        tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/{llm_name}_{output_col}_fine_tuned')

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id
    korean_prompter = Prompter('korean')

    # run inference using Fine-Tuned LLM
    for final_input_prompt in valid_final_input_prompts:
        print(f'\nLLM input :\n{final_input_prompt}')

        # generate 4 answers for comparison
        llm_answers = []
        trial_counts = []
        output_token_cnts = []

        for _ in range(4):
            llm_answer, trial_count, output_token_cnt = None, None, None

            if llm_name == 'koreanlm':
                llm_answer, trial_count, output_token_cnt = run_inference_koreanlm(fine_tuned_llm,
                                                                                   final_input_prompt,
                                                                                   tokenizer,
                                                                                   prompter=korean_prompter)

            elif llm_name == 'polyglot':
                llm_answer, trial_count, output_token_cnt = run_inference(fine_tuned_llm,
                                                                          final_input_prompt,
                                                                          tokenizer,
                                                                          stop_token_list=[1477, 1078, 4833, 12],
                                                                          answer_start_mark=' (답변 시작)',
                                                                          remove_token_type_ids=True)

            llm_answers.append(llm_answer)
            trial_counts.append(str(trial_count))
            output_token_cnts.append(str(output_token_cnt))

        trial_counts_str = ','.join(trial_counts)
        output_token_cnts_str = ','.join(output_token_cnts)
        llm_answers_str = '\n- '.join(llm_answers)

        print(f'Oh-LoRA answer (trials: {trial_counts_str} | output_tkn_cnt : {output_token_cnts_str}) '
              f':\n- {llm_answers_str}')
