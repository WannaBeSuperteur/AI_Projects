import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # remove for LLM inference only

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tuning.fine_tuning_polyglot import fine_tune_model as fine_tune_polyglot
from fine_tuning.fine_tuning_polyglot import get_stop_token_list as get_stop_token_list_polyglot
from fine_tuning.inference import run_inference_polyglot

from fine_tuning.fine_tuning_kanana import fine_tune_model as fine_tune_kanana
from fine_tuning.fine_tuning_kanana import get_stop_token_list as get_stop_token_list_kanana
from fine_tuning.inference import run_inference_kanana

from fine_tuning.utils import load_valid_final_prompts, get_answer_start_mark, get_temperature


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Fine-Tuning 된 LLM 로딩
# Create Date : 2025.05.31
# Last Update Date : -

# Arguments:
# - llm_name   (str) : Fine-Tuning 된 LLM 의 이름 ('kanana' or 'polyglot')
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 LLM

def load_fine_tuned_llm(llm_name, output_col):
    fine_tuned_llm = None

    if llm_name == 'kanana':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/llm/models/kanana_{output_col}_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    if llm_name == 'polyglot':
        fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/llm/models/polyglot_{output_col}_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

    return fine_tuned_llm


# LLM 4개 한번에 로딩 시 OOM 발생 테스트 (Polyglot-Ko 1.3B)
# Create Date : 2025.05.31
# Last Update Date : -

# Arguments:
# - is_separate (bool) : True 이면 2 대의 GPU 분산 로딩, False 이면 1 대의 GPU 에만 로딩

def test_cuda_oom_polyglot(is_separate):
    gpu_0 = torch.device('cuda:0')
    gpu_1 = torch.device('cuda:1')

    output_cols = ['output_message', 'memory', 'eyes_mouth_pose', 'summary']

    llms = {}
    if is_separate:
        device_mapping = {'output_message': gpu_0, 'memory': gpu_0, 'eyes_mouth_pose': gpu_1, 'summary': gpu_1}
    else:
        device_mapping = {'output_message': gpu_0, 'memory': gpu_0, 'eyes_mouth_pose': gpu_0, 'summary': gpu_0}

    for col in output_cols:
        llms[col] = AutoModelForCausalLM.from_pretrained(
            f'{PROJECT_DIR_PATH}/llm/models/polyglot_{col}_fine_tuned',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).to(device_mapping[col])

        llm_tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/{llm_name}_{col}_fine_tuned')
        print(f'\nCUDA OOM test for Fine-Tuned LLM ({llm_name}, {col}) - Load SUCCESSFUL! 👱‍♀️')

        llms[col].generation_config.pad_token_id = llm_tokenizer.pad_token_id
        input_prompt = '로라야 사랑해! 요즘 잘 지내고 있어?'

        stop_token_list = get_stop_token_list_polyglot(col)

        llm_output, _, _ = run_inference_polyglot(llms[col],
                                                  input_prompt,
                                                  llm_tokenizer,
                                                  col,
                                                  stop_token_list=stop_token_list,
                                                  answer_start_mark=get_answer_start_mark(col))

        print(f'LLM output : {llm_output}')
        print(f'CUDA memory until {col} model loading : {torch.cuda.memory_allocated()}')


# LLM Fine-Tuning 학습 실시
# Create Date : 2025.05.31
# Last Update Date : -

# Arguments:
# - llm_name   (str) : Fine-Tuning 할 LLM 의 이름 ('kanana' or 'polyglot')
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

def fine_tune_llm(llm_name, output_col):

    # load valid dataset
    valid_final_input_prompts = load_valid_final_prompts(output_col=output_col)

    for final_input_prompt in valid_final_input_prompts:
        print(f'final input prompt for validation : {final_input_prompt}')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm(llm_name, output_col)
        tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/{llm_name}_{output_col}_fine_tuned')
        print(f'Fine-Tuned LLM ({llm_name}) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'Fine-Tuned LLM ({llm_name}) load failed : {e}')

        if llm_name == 'kanana':
            fine_tune_kanana(output_col=output_col, dataset_version='v2_2')

        elif llm_name == 'polyglot':
            fine_tune_polyglot(output_col=output_col, dataset_version='v2_2')

        fine_tuned_llm = load_fine_tuned_llm(llm_name, output_col)
        tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/{llm_name}_{output_col}_fine_tuned')

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    inference_temperature = get_temperature(output_col, llm_name)
    llm_log_path = f'{PROJECT_DIR_PATH}/llm/fine_tuning/logs'
    inference_log_path = f'{llm_log_path}/{llm_name}_{output_col}_inference_log_{inference_temperature}.txt'
    inference_log = ''

    # run inference using Fine-Tuned LLM
    for final_input_prompt in valid_final_input_prompts:
        llm_input_print = f'\nLLM input :\n{final_input_prompt}'
        print(llm_input_print)

        # generate 4 answers for comparison
        llm_answers = []
        trial_counts = []
        output_token_cnts = []

        for _ in range(4):
            llm_answer, trial_count, output_token_cnt = None, None, None
            answer_start_mark = get_answer_start_mark(output_col)

            if llm_name == 'kanana':
                stop_token_list = get_stop_token_list_kanana(output_col)

                llm_answer, trial_count, output_token_cnt = run_inference_kanana(fine_tuned_llm,
                                                                                 final_input_prompt,
                                                                                 tokenizer,
                                                                                 output_col,
                                                                                 stop_token_list=stop_token_list,
                                                                                 answer_start_mark=answer_start_mark)

            if llm_name == 'polyglot':
                stop_token_list = get_stop_token_list_polyglot(output_col)

                llm_answer, trial_count, output_token_cnt = run_inference_polyglot(fine_tuned_llm,
                                                                                   final_input_prompt,
                                                                                   tokenizer,
                                                                                   output_col,
                                                                                   stop_token_list=stop_token_list,
                                                                                   answer_start_mark=answer_start_mark)

            llm_answers.append(llm_answer)
            trial_counts.append(str(trial_count))
            output_token_cnts.append(str(output_token_cnt))

        trial_counts_str = ','.join(trial_counts)
        output_token_cnts_str = ','.join(output_token_cnts)
        llm_answers_str = '\n- '.join(llm_answers)

        llm_output_print = (f'Oh-LoRA answer (trials: {trial_counts_str} | output_tkn_cnt : {output_token_cnts_str}) '
                            f':\n- {llm_answers_str}')
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
                        default='kanana,polyglot,kanana,polyglot')

    parser.add_argument('-output_cols',
                        help="output column names (separated by comma) from dataset csv",
                        default='output_message,memory,summary,eyes_mouth_pose')

    args = parser.parse_args()

    llm_names = args.llm_names
    output_cols = args.output_cols

    # CUDA OOM test
    # Polyglot-Ko 1.3B : (separated = False -> Result : max         11271 MiB / 12288 MiB -> 2대의 GPU 에 분산 로딩 필요)
    #                    (separated = True  -> Result : max  (6060, 5349) MiB / 12288 MiB)
#    test_cuda_oom_polyglot(True)

    llm_names_list = llm_names.split(',')
    output_cols_list = output_cols.split(',')

    for llm_name, output_col in zip(llm_names_list, output_cols_list):
        assert llm_name in ['kanana', 'polyglot'], "LLM name must be 'kanana' or 'polyglot'."

        print(f'\n=== 🚀 Fine-Tune LLM {llm_name} with column {output_col} START 🚀 \n===')
        fine_tune_llm(llm_name, output_col)
