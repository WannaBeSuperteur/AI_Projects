import os
import time

import peft
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainingArguments, TrainerState, \
                         TrainerControl

import torch
import pandas as pd

try:
    from fine_tuning.inference import run_inference_kanana
    from fine_tuning.utils import load_valid_final_prompts, preview_dataset, add_train_log, add_inference_log, \
                                  get_answer_start_mark
    from fine_tuning.augmentation import AugmentCollator
except:
    from llm.fine_tuning.inference import run_inference_kanana
    from llm.fine_tuning.utils import load_valid_final_prompts, preview_dataset, add_train_log, add_inference_log, \
        get_answer_start_mark
    from llm.fine_tuning.augmentation import AugmentCollator


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

lora_llm = None
tokenizer = None
valid_final_prompts = None

train_log_dict = {'epoch': [], 'time': [], 'loss': [], 'grad_norm': [], 'learning_rate': [], 'mean_token_accuracy': []}
inference_log_dict = {'epoch': [], 'elapsed_time (s)': [], 'prompt': [], 'llm_answer': [],
                      'trial_cnt': [], 'output_tkn_cnt': []}

log_dir_path = f'{PROJECT_DIR_PATH}/llm/fine_tuning/logs'
os.makedirs(log_dir_path, exist_ok=True)


def get_stop_token_list(output_col):
    if output_col == 'output_message':
        return [109659, 104449, 99458, 64356, 8, 220]  # (답변 종료)
    elif output_col == 'summary':
        return [36811, 103168, 99458, 64356, 8, 220]  # (요약 종료)
    elif output_col == 'memory':
        return [36811, 103168, 99458, 64356, 8, 220]  # (요약 종료)
    else:  # eyes_mouth_pose
        return [320, 102260, 30381, 62226, 99458, 64356, 8, 220]  # (표정 출력 종료)


class OhLoRACustomCallback(TrainerCallback):

    def __init__(self, output_col, instruct_version):
        super(OhLoRACustomCallback, self).__init__()
        self.output_col = output_col
        self.instruct_version = instruct_version

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_final_prompts

        kanana_llm_name = 'kananai' if self.instruct_version else 'kanana'
        train_log_df = pd.DataFrame(train_log_dict)
        train_log_df.to_csv(f'{log_dir_path}/{kanana_llm_name}_{self.output_col}_train_log.csv')

        print('=== INFERENCE TEST ===')

        for final_input_prompt in valid_final_prompts:
            start_at = time.time()
            stop_token_list = get_stop_token_list(self.output_col)
            answer_start_mark = get_answer_start_mark(self.output_col)

            llm_answer, trial_count, output_token_cnt = run_inference_kanana(lora_llm,
                                                                             final_input_prompt,
                                                                             tokenizer,
                                                                             output_col=self.output_col,
                                                                             stop_token_list=stop_token_list,
                                                                             answer_start_mark=answer_start_mark,
                                                                             instruct_version=self.instruct_version)
            elapsed_time = time.time() - start_at

            print(f'final input prompt : {final_input_prompt}')
            print(f'llm answer (trials: {trial_count}, output tkns: {output_token_cnt}) : {llm_answer}')

            inference_result = {'epoch': state.epoch, 'elapsed_time': elapsed_time, 'prompt': final_input_prompt,
                                'llm_answer': llm_answer, 'trial_cnt': trial_count, 'output_tkn_cnt': output_token_cnt}
            add_inference_log(inference_result, inference_log_dict)

        inference_log_df = pd.DataFrame(inference_log_dict)
        inference_log_df.to_csv(f'{log_dir_path}/{kanana_llm_name}_{self.output_col}_inference_log_dict.csv')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            add_train_log(state, train_log_dict)
        except Exception as e:
            print(f'logging failed : {e}')


# Original LLM (Kanana-1.5 2.1B) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2025.05.31
# Last Update Date : 2025.06.04
# - Kanana-1.5-2.1B instruct LLM (kananai) 옵션 추가

# Arguments:
# - kanana_llm_name (str) : 'kananai' for Kanana-1.5-2.1B instruct, 'kanana' for Kanana-1.5-2.1B base

# Returns:
# - original_llm (LLM) : Original Kanana-1.5 2.1B LLM

def get_original_llm(kanana_llm_name):
    original_llm = AutoModelForCausalLM.from_pretrained(
        f'{PROJECT_DIR_PATH}/llm/models/{kanana_llm_name}_original',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).cuda()

    return original_llm


# Original LLM (Kanana-1.5 2.1B) 에 대한 Fine-Tuning 을 위한 Training Arguments 가져오기
# Create Date : 2025.05.31
# Last Update Date : 2025.06.04
# - Kanana-1.5-2.1B instruct LLM (kananai) 옵션 추가에 따른 kanana_llm_name 변수 추가

# Arguments:
# - output_col      (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name
# - kanana_llm_name (str) : 'kananai' for Kanana-1.5-2.1B instruct, 'kanana' for Kanana-1.5-2.1B base

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args(output_col, kanana_llm_name):
    output_dir_path = f'{PROJECT_DIR_PATH}/llm/models/{kanana_llm_name}_{output_col}_fine_tuned'
    num_train_epochs_dict = {'output_message': 5, 'memory': 20, 'summary': 10, 'eyes_mouth_pose': 30}
    num_train_epochs = num_train_epochs_dict[output_col]

    training_args = SFTConfig(
        learning_rate=0.0003,               # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=num_train_epochs,
        logging_steps=20,                   # logging frequency
        gradient_checkpointing=False,
        output_dir=output_dir_path,
        save_total_limit=3,                 # max checkpoint count to save
        per_device_train_batch_size=2,      # batch size per device during training
        per_device_eval_batch_size=1,       # batch size per device during validation
        report_to=None                      # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


# Original LLM (Kanana-1.5 2.1B) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2025.05.31
# Last Update Date : 2025.06.04
# - Kanana-1.5-2.1B instruct LLM (kananai) 옵션 추가에 따른 instruct_version 변수 추가

# Arguments:
# - dataset          (Dataset)      : LLM 학습 데이터셋
# - collator         (DataCollator) : Data Collator
# - training_args    (SFTConfig)    : Training Arguments
# - output_col       (str)          : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name
# - instruct_version (bool)         : True for Kanana-1.5-2.1B instruct, False for Kanana-1.5-2.1B base

# Returns:
# - trainer (SFTTrainer) : SFT (Supervised Fine-Tuning) Trainer

def get_sft_trainer(dataset, collator, training_args, output_col, instruct_version):
    global lora_llm, tokenizer

    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,     # LLM tokenizer / renamed : tokenizer -> processing_class from trl 0.12.0
        args=training_args,
        data_collator=collator,
        callbacks=[OhLoRACustomCallback(output_col, instruct_version)]
    )

    return trainer


# Original LLM (Kanana-1.5 2.1B) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.05.31
# Last Update Date : -

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (Kanana-1.5 2.1B)
# - lora_rank (int) : LoRA 적용 시의 Rank

# Returns:
# - lora_llm (LLM) : LoRA 가 적용된 LLM

def get_lora_llm(llm, lora_rank):
    global lora_llm

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

    lora_llm = peft.get_peft_model(llm, lora_config)
    lora_llm.print_trainable_parameters()


# Original LLM (Kanana-1.5 2.1B) 에 대한 LLM 이 직접 학습 가능한 데이터셋 가져오기
# Create Date : 2025.05.31
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame
#                                   (from llm/fine_tuning_dataset/OhLoRA_fine_tuning_{v2_2|v3}.csv)
#                                   columns = ['data_type', 'input_data', ...]

# Returns:
# - dataset (Dataset) : LLM 학습 데이터셋

def generate_llm_trainable_dataset(dataset_df):
    global tokenizer

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'train'][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'valid'][['text']])
    preview_dataset(dataset, tokenizer)

    return dataset


# LLM (Kanana-1.5 2.1B) Fine-Tuning 실시
# Create Date : 2025.05.31
# Last Update Date : 2025.06.04
# - Kanana-1.5-2.1B instruct LLM (kananai) 옵션 추가

# Arguments:
# - output_col       (str)  : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name
# - dataset_version  (str)  : 학습 데이터셋 버전 ('v2_2' or 'v3')
# - instruct_version (bool) : True for Kanana-1.5-2.1B instruct, False for Kanana-1.5-2.1B base

# Returns:
# - llm/models/kanana_{output_col}_fine_tuned 또는 llm/models/kananai_{output_col}_fine_tuned 에 Fine-Tuning 된 모델 저장

def fine_tune_model(output_col, dataset_version, instruct_version):
    global lora_llm, tokenizer, valid_final_prompts
    valid_final_prompts = load_valid_final_prompts(output_col=output_col)
    kanana_llm_name = 'kananai' if instruct_version else 'kanana'

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    # Kanana-1.5 2.1B original model is from https://huggingface.co/kakaocorp/kanana-1.5-2.1b-base (base)
    #                                     or https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505 (instruct)
    original_llm = get_original_llm(kanana_llm_name)
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/{kanana_llm_name}_original')

    tokenizer.pad_token = tokenizer.eos_token
    original_llm.generation_config.pad_token_id = tokenizer.pad_token_id  # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

    # read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/fine_tuning_dataset/OhLoRA_fine_tuning_{dataset_version}.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=original_llm, lora_rank=16)

    if output_col == 'output_message':
        dataset_df['text'] = dataset_df.apply(
            lambda x: f"{x['input_data']} (답변 시작) ### 답변: {x[output_col]} (답변 종료) <|end_of_text|>",
            axis=1)

    elif output_col == 'summary':
        dataset_df['text'] = dataset_df.apply(
            lambda x: f"{x['input_data'] + ' (오로라 답변) ' + x['output_message']} (요약 시작) ### 답변: {x[output_col]} (요약 종료) <|end_of_text|>",
            axis=1)

    elif output_col == 'memory':
        dataset_df['text'] = dataset_df.apply(
            lambda x: f"{x['input_data']} (요약 시작) ### 답변: {'' if str(x[output_col]) == 'nan' else x[output_col]} (요약 종료) <|end_of_text|>",
            axis=1)

    else:  # eyes_mouth_pose
        dataset_df['text'] = dataset_df.apply(
            lambda x: f"{x['output_message']} (표정 출력 시작) ### 답변: {x[output_col]} (표정 출력 종료) <|end_of_text|>",
            axis=1)

    dataset = generate_llm_trainable_dataset(dataset_df)
    preview_dataset(dataset, tokenizer)

    response_template = [8, 17010, 111964, 25]  # '### 답변 :'

    if output_col == 'output_message':
        collator = AugmentCollator(response_template, llm_name=kanana_llm_name, tokenizer=tokenizer)
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = get_training_args(output_col, kanana_llm_name)
    trainer = get_sft_trainer(dataset, collator, training_args, output_col, instruct_version)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    output_dir_path = f'{PROJECT_DIR_PATH}/llm/models/{kanana_llm_name}_{output_col}_fine_tuned'
    trainer.save_model(output_dir_path)
