
from fine_tuning.fine_tuning_kanana import get_original_llm, generate_llm_trainable_dataset, get_training_args,\
                                           get_stop_token_list
from fine_tuning.augmentation import AugmentCollator
from fine_tuning.utils import preview_dataset, get_answer_start_mark, load_valid_final_prompts
from fine_tuning.inference import run_inference_kanana
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from trl import SFTTrainer

import peft
from peft import LoraConfig

import numpy as np
import pandas as pd

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # remove for LLM inference only
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
EXPERIMENT_TIMES = 5

kanana_llm = None
tokenizer = None
valid_final_prompts = load_valid_final_prompts(output_col='output_message')

entire_log = {'llm_name': [], 'epochs': [], 'elapsed_time_mean': [], 'output_tokens_mean': [], 'num_start_answers': []}
test_log_dir_path = f'{PROJECT_DIR_PATH}/llm/fine_tuning/logs_kanana_base_vs_instruct'
test_log_csv_path = f'{test_log_dir_path}/experiment_log.csv'
os.makedirs(test_log_dir_path, exist_ok=True)


class BaseVsInstructCustomCallback(TrainerCallback):

    def __init__(self, output_col, instruct_version):
        super(BaseVsInstructCustomCallback, self).__init__()
        self.output_col = output_col
        self.instruct_version = instruct_version

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global kanana_llm, tokenizer, valid_final_prompts, entire_log

        elapsed_time_log = []
        output_tokens_log = []
        llm_answer_starts_with_num = 0

        kanana_llm_name = 'kananai' if self.instruct_version else 'kanana'
        print('=== INFERENCE TEST ===')

        for idx, final_input_prompt in enumerate(valid_final_prompts):
            start_at = time.time()
            stop_token_list = get_stop_token_list(self.output_col)
            answer_start_mark = get_answer_start_mark(self.output_col)

            llm_answer, trial_count, output_token_cnt = run_inference_kanana(kanana_llm,
                                                                             final_input_prompt,
                                                                             tokenizer,
                                                                             output_col=self.output_col,
                                                                             stop_token_list=stop_token_list,
                                                                             answer_start_mark=answer_start_mark,
                                                                             instruct_version=self.instruct_version)
            print(f'valid dataset idx : {idx}, llm answer : {llm_answer}')

            elapsed_time = time.time() - start_at
            elapsed_time_log.append(elapsed_time)
            output_tokens_log.append(output_token_cnt)

            if llm_answer.replace(' ', '')[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                llm_answer_starts_with_num += 1

        entire_log['llm_name'].append(kanana_llm_name)
        entire_log['epochs'].append(int(round(state.log_history[-1]['epoch'])))
        entire_log['elapsed_time_mean'].append(np.mean(elapsed_time_log))
        entire_log['output_tokens_mean'].append(np.mean(output_tokens_log))
        entire_log['num_start_answers'].append(llm_answer_starts_with_num)

        pd.DataFrame(entire_log).to_csv('')


# Original LLM (Kanana-1.5 2.1B) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.06.04
# Last Update Date : -

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (Kanana-1.5 2.1B)
# - lora_rank (int) : LoRA 적용 시의 Rank

# Returns:
# - lora_llm (LLM) : LoRA 가 적용된 LLM

def get_lora_llm(llm, lora_rank):
    global kanana_llm

    # Kanana-1.5 is based on LlamaForCausalLM architecture
    # target modules of LlamaForCausalLM : ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_rank,                          # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,                    # Dropout for LoRA
        init_lora_weights="gaussian",         # LoRA weight initialization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    kanana_llm = peft.get_peft_model(llm, lora_config)
    kanana_llm.print_trainable_parameters()


# Original LLM (Kanana-1.5 2.1B) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2025.06.04
# Last Update Date : -

# Arguments:
# - dataset          (Dataset)      : LLM 학습 데이터셋
# - collator         (DataCollator) : Data Collator
# - training_args    (SFTConfig)    : Training Arguments
# - output_col       (str)          : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name
# - instruct_version (bool)         : True for Kanana-1.5-2.1B instruct, False for Kanana-1.5-2.1B base

# Returns:
# - trainer (SFTTrainer) : SFT (Supervised Fine-Tuning) Trainer

def get_sft_trainer(dataset, collator, training_args, output_col, instruct_version):
    global kanana_llm, tokenizer

    trainer = SFTTrainer(
        kanana_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,     # LLM tokenizer / renamed : tokenizer -> processing_class from trl 0.12.0
        args=training_args,
        data_collator=collator,
        callbacks=[BaseVsInstructCustomCallback(output_col, instruct_version)]
    )

    return trainer


# Kanana LLM (Base or Instruct) Fine-Tuning & inference 실험
# Create Date : 2025.06.04
# Last Update Date : -

# Arguments:
# - kanana_llm_name  (str)  : 'kananai' for Kanana-1.5-2.1B instruct, 'kanana' for Kanana-1.5-2.1B base
# - instruct_version (bool) : True for Kanana-1.5-2.1B instruct, False for Kanana-1.5-2.1B base

def fine_tune_kanana_llm(kanana_llm_name, instruct_version):
    global kanana_llm, tokenizer

    kanana_llm = get_original_llm(kanana_llm_name)
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/{kanana_llm_name}_original')

    tokenizer.pad_token = tokenizer.eos_token
    kanana_llm.generation_config.pad_token_id = tokenizer.pad_token_id  # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

    # read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/fine_tuning_dataset/OhLoRA_fine_tuning_v3.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=kanana_llm, lora_rank=64)

    dataset_df['text'] = dataset_df.apply(
        lambda x: f"{x['input_data']} (답변 시작) ### 답변: {x['output_message']} (답변 종료) <|end_of_text|>",
        axis=1)

    dataset = generate_llm_trainable_dataset(dataset_df)
    preview_dataset(dataset, tokenizer)

    response_template = [8, 17010, 111964, 25]  # '### 답변 :'
    collator = AugmentCollator(response_template, llm_name=kanana_llm_name, tokenizer=tokenizer)

    training_args = get_training_args('output_message', kanana_llm_name)
    trainer = get_sft_trainer(dataset, collator, training_args, 'output_message', instruct_version)

    # run Fine-Tuning
    trainer.train()


if __name__ == '__main__':
    kanana_llm_names = ['kanana', 'kananai']

    for kanana_llm_name in kanana_llm_names:
        instruct_version = (kanana_llm_name == 'kananai')

        for exp_idx in range(EXPERIMENT_TIMES):
            fine_tune_kanana_llm(kanana_llm_name, instruct_version=instruct_version)
