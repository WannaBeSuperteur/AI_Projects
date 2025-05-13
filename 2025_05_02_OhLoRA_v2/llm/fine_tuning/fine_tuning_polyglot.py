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

from fine_tuning.inference import run_inference
from fine_tuning.utils import load_valid_final_prompts, preview_dataset, add_train_log, add_inference_log


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

lora_llm = None
tokenizer = None
valid_final_prompts = None

train_log_dict = {'epoch': [], 'time': [], 'loss': [], 'grad_norm': [], 'learning_rate': [], 'mean_token_accuracy': []}
inference_log_dict = {'epoch': [], 'elapsed_time (s)': [], 'prompt': [], 'llm_answer': [],
                      'trial_cnt': [], 'output_tkn_cnt': []}

log_dir_path = f'{PROJECT_DIR_PATH}/llm/fine_tuning/logs'
os.makedirs(log_dir_path, exist_ok=True)


class OhLoRACustomCallback(TrainerCallback):

    def __init__(self, output_col):
        super(OhLoRACustomCallback, self).__init__()
        self.output_col = output_col

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_final_prompts

        train_log_df = pd.DataFrame(train_log_dict)
        train_log_df.to_csv(f'{log_dir_path}/polyglot_{self.output_col}_train_log.csv')

        print('=== INFERENCE TEST ===')

        for final_input_prompt in valid_final_prompts:
            start_at = time.time()
            llm_answer, trial_count, output_token_cnt = run_inference(lora_llm,
                                                                      final_input_prompt,
                                                                      tokenizer,
                                                                      stop_token_list=[1477, 1078, 4833, 12],
                                                                      answer_start_mark=' (답변 시작)',
                                                                      remove_token_type_ids=True)
            elapsed_time = time.time() - start_at

            print(f'final input prompt : {final_input_prompt}')
            print(f'llm answer (trials: {trial_count}, output tkns: {output_token_cnt}) : {llm_answer}')

            inference_result = {'epoch': state.epoch, 'elapsed_time': elapsed_time, 'prompt': final_input_prompt,
                                'llm_answer': llm_answer, 'trial_cnt': trial_count, 'output_tkn_cnt': output_token_cnt}
            add_inference_log(inference_result, inference_log_dict)

        inference_log_df = pd.DataFrame(inference_log_dict)
        inference_log_df.to_csv(f'{log_dir_path}/polyglot_{self.output_col}_inference_log_dict.csv')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            add_train_log(state, train_log_dict)
        except Exception as e:
            print(f'logging failed : {e}')


# Original LLM (Polyglot-Ko 1.3B) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2025.05.12
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - original_llm (LLM) : Original Polyglot-Ko 1.3B LLM

def get_original_llm():
    original_llm = AutoModelForCausalLM.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/polyglot_original',
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16).cuda()

    return original_llm


# Original LLM (Polyglot-Ko 1.3B) 에 대한 Fine-Tuning 을 위한 Training Arguments 가져오기
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - 업데이트된 학습 데이터셋 (OhLoRA_fine_tuning_v2.csv) 반영 및 총 4 개의 LLM 개별 학습
# - LLM output column 에 따라 서로 다른 training argument 적용

# Arguments:
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args(output_col):
    output_dir_path = f'{PROJECT_DIR_PATH}/llm/models/polyglot_{output_col}_fine_tuned'
    num_train_epochs_dict = {'output_message': 80}
    num_train_epochs = num_train_epochs_dict[output_col]

    training_args = SFTConfig(
        learning_rate=0.0002,               # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=num_train_epochs,
        logging_steps=10,                   # logging frequency
        gradient_checkpointing=False,
        output_dir=output_dir_path,
        save_total_limit=3,                 # max checkpoint count to save
        per_device_train_batch_size=4,      # batch size per device during training
        per_device_eval_batch_size=1,       # batch size per device during validation
        report_to=None                      # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


# Original LLM (Polyglot-Ko 1.3B) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - callback 함수 이름 수정 및 인수 추가

# Arguments:
# - dataset       (Dataset)      : LLM 학습 데이터셋
# - collator      (DataCollator) : Data Collator
# - training_args (SFTConfig)    : Training Arguments
# - output_col    (str)          : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - trainer (SFTTrainer) : SFT (Supervised Fine-Tuning) Trainer

def get_sft_trainer(dataset, collator, training_args, output_col):
    global lora_llm, tokenizer

    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,     # LLM tokenizer / renamed : tokenizer -> processing_class from trl 0.12.0
        args=training_args,
        data_collator=collator,
        callbacks=[OhLoRACustomCallback(output_col)]
    )

    return trainer


# Original LLM (Polyglot-Ko 1.3B) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.05.12
# Last Update Date : -

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (Polyglot-Ko 1.3B)
# - lora_rank (int) : LoRA 적용 시의 Rank

# Returns:
# - lora_llm (LLM) : LoRA 가 적용된 LLM

def get_lora_llm(llm, lora_rank):
    global lora_llm

    lora_config = LoraConfig(
        r=lora_rank,                          # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,                    # Dropout for LoRA
        init_lora_weights="gaussian",         # LoRA weight initialization
        target_modules=["query_key_value"],
        task_type="CAUSAL_LM"
    )

    lora_llm = peft.get_peft_model(llm, lora_config)
    lora_llm.print_trainable_parameters()


# Original LLM (Polyglot-Ko 1.3B) 에 대한 LLM 이 직접 학습 가능한 데이터셋 가져오기
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - train data preview 출력 삭제

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame (from llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv)
#                                   columns = ['data_type', 'input_data', ...]

# Returns:
# - dataset (Dataset) : LLM 학습 데이터셋

def generate_llm_trainable_dataset(dataset_df):
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'train'][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'valid'][['text']])

    return dataset


# LLM (Polyglot-Ko 1.3B) Fine-Tuning 실시
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - 업데이트된 학습 데이터셋 (OhLoRA_fine_tuning_v2.csv) 반영 및 총 4 개의 LLM 개별 학습
# - dataset preview 추가

# Arguments:
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - 2025_05_02_OhLoRA_v2/llm/models/polyglot_{output_col}_fine_tuned 에 Fine-Tuning 된 모델 저장

def fine_tune_model(output_col):
    global lora_llm, tokenizer, valid_final_prompts
    valid_final_prompts = load_valid_final_prompts(dataset_csv_path='llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv',
                                                   output_col=output_col)

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    # Polyglot-Ko original model is from https://huggingface.co/EleutherAI/polyglot-ko-1.3b
    original_llm = get_original_llm()
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/polyglot_original')
    original_llm.generation_config.pad_token_id = tokenizer.pad_token_id  # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

    # read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=original_llm, lora_rank=128)

#    print(tokenizer.encode('### 답변:'))  # ... [6, 6, 6, 4253, 29]
#    print(tokenizer.encode('(답변 시작) ### 답변:'))  # ... [11, 1477, 1078, 1016, 12, 6501, 6, 6, 4253, 29]
#    print(tokenizer.encode('(답변 종료)'))  # ... [11, 1477, 1078, 4833, 12]

    if output_col == 'summary':
        dataset_df['text'] = dataset_df.apply(lambda x: f"{x['input_data'] + ' / ' + x['output_message']} (답변 시작) ### 답변: {x[output_col]} (답변 종료) <|endoftext|>",
                                              axis=1)

    else:
        dataset_df['text'] = dataset_df.apply(lambda x: f"{x['input_data']} (답변 시작) ### 답변: {x[output_col]} (답변 종료) <|endoftext|>",
                                              axis=1)

    dataset = generate_llm_trainable_dataset(dataset_df)
    preview_dataset(dataset)

    response_template = [6, 6, 4253, 29]  # '### 답변 :'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = get_training_args(output_col)
    trainer = get_sft_trainer(dataset, collator, training_args, output_col)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    output_dir_path = f'{PROJECT_DIR_PATH}/llm/models/polyglot_{output_col}_fine_tuned'
    trainer.save_model(output_dir_path)
