import os
import time

import peft
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainingArguments, TrainerState, \
                         TrainerControl

import torch
import pandas as pd

try:
    from llm_fine_tuning.inference import run_inference_midm
    from llm_fine_tuning.utils import load_valid_final_prompts, preview_dataset, add_train_log, add_inference_log, \
                                      get_answer_start_mark, convert_into_filled_df

except:
    from ai_interview.llm_fine_tuning.inference import run_inference_midm
    from ai_interview.llm_fine_tuning.utils import load_valid_final_prompts, preview_dataset, add_train_log, \
                                                   add_inference_log, get_answer_start_mark, convert_into_filled_df


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

lora_llm = None
tokenizer = None
valid_final_prompts = None

train_log_dict = {'epoch': [], 'time': [], 'loss': [], 'grad_norm': [], 'learning_rate': [], 'mean_token_accuracy': []}
inference_log_dict = {'epoch': [], 'elapsed_time (s)': [], 'prompt': [], 'llm_answer': [],
                      'trial_cnt': [], 'output_tkn_cnt': []}

log_dir_path = f'{PROJECT_DIR_PATH}/ai_interview/llm_fine_tuning/logs'
os.makedirs(log_dir_path, exist_ok=True)


def get_stop_token_list():
    return [663, 425, 4511]  # (해설 종료)


class OhLoRACustomCallback(TrainerCallback):

    def __init__(self, epochs):
        super(OhLoRACustomCallback, self).__init__()
        self.epochs = epochs

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_final_prompts

        train_log_df = pd.DataFrame(train_log_dict)
        train_log_df.to_csv(f'{log_dir_path}/midm_sft_final_train_log_{self.epochs}epochs.csv')

        print('=== INFERENCE TEST ===')

        for final_input_prompt in valid_final_prompts:
            start_at = time.time()
            stop_token_list = get_stop_token_list()
            answer_start_mark = get_answer_start_mark()

            llm_answer, trial_count, output_token_cnt = run_inference_midm(lora_llm,
                                                                           final_input_prompt,
                                                                           tokenizer,
                                                                           stop_token_list=stop_token_list,
                                                                           answer_start_mark=answer_start_mark)
            elapsed_time = time.time() - start_at

            print(f'final input prompt : {final_input_prompt}')
            print(f'llm answer (trials: {trial_count}, output tkns: {output_token_cnt}) : {llm_answer}\n')

            inference_result = {'epoch': state.epoch, 'elapsed_time': elapsed_time, 'prompt': final_input_prompt,
                                'llm_answer': llm_answer, 'trial_cnt': trial_count, 'output_tkn_cnt': output_token_cnt}
            add_inference_log(inference_result, inference_log_dict)

        inference_log_df = pd.DataFrame(inference_log_dict)
        inference_log_df.to_csv(f'{log_dir_path}/midm_sft_final_inference_log_dict_{self.epochs}epochs.csv')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            add_train_log(state, train_log_dict)
        except Exception as e:
            print(f'logging failed : {e}')


# Original LLM (Mi:dm 2.0 Mini) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - original_llm (LLM) : Original Mi:dm 2.0 Mini (2.31B) LLM

def get_original_llm():
    original_llm_path = f'{PROJECT_DIR_PATH}/llm_original_models/midm_original'

    original_llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=original_llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).cuda()

    print(f'Original LLM load successful : {original_llm_path}')
    return original_llm


# Original LLM (Mi:dm 2.0 Mini) 에 대한 Fine-Tuning 을 위한 Training Arguments 가져오기
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# -  epochs (int) : 학습 epoch 횟수

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args(epochs):
    output_dir_path = f'{PROJECT_DIR_PATH}/ai_interview/models/midm_sft_final_fine_tuned_{epochs}epochs'

    training_args = SFTConfig(
        learning_rate=0.0003,               # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=epochs,
        logging_steps=5,                    # logging frequency
        gradient_checkpointing=False,
        output_dir=output_dir_path,
        save_total_limit=3,                 # max checkpoint count to save
        per_device_train_batch_size=1,      # batch size per device during training
        per_device_eval_batch_size=1,       # batch size per device during validation
        report_to=None                      # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


# Original LLM (Mi:dm 2.0 Mini) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - dataset       (Dataset)      : LLM 학습 데이터셋
# - collator      (DataCollator) : Data Collator
# - training_args (SFTConfig)    : Training Arguments
# - epochs        (int)          : 학습 epoch 횟수

# Returns:
# - trainer (SFTTrainer) : SFT (Supervised Fine-Tuning) Trainer

def get_sft_trainer(dataset, collator, training_args, epochs):
    global lora_llm, tokenizer

    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,     # LLM tokenizer / renamed : tokenizer -> processing_class from trl 0.12.0
        args=training_args,
        data_collator=collator,
        callbacks=[OhLoRACustomCallback(epochs)]
    )

    return trainer


# Original LLM (Mi:dm 2.0 Mini) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (Mi:dm 2.0 Mini, 2.31B)
# - lora_rank (int) : LoRA 적용 시의 Rank

# Returns:
# - lora_llm (LLM) : LoRA 가 적용된 LLM

def get_lora_llm(llm, lora_rank):
    global lora_llm

    # Mi:dm 2.0 Mini is based on LlamaForCausalLM architecture
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


# Original LLM (Mi:dm 2.0 Mini) 에 대한 LLM 이 직접 학습 가능한 데이터셋 가져오기
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame (from ai_interview/dataset/all_train_and_test_data.csv)
#                                   columns = ['data_type', 'input_data_wo_rag_augment', 'user_input', ...]

# Returns:
# - dataset (Dataset) : LLM 학습 데이터셋

def generate_llm_trainable_dataset(dataset_df):
    global tokenizer

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'].str.startswith('train')][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'].str.startswith('valid')][['text']])
    preview_dataset(dataset, tokenizer)

    return dataset


# LLM (Mi:dm 2.0 Mini) Fine-Tuning 실시
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - epochs (int) : 학습 epoch 횟수

# Returns:
# - ai_qna/models/midm_sft_final_fine_tuned 에 Fine-Tuning 된 모델 저장

def fine_tune_model(epochs):
    global lora_llm, tokenizer, valid_final_prompts
    valid_final_prompts = load_valid_final_prompts()

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    # Mi:dm 2.0 Mini (2.31B) original model is from https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct
    original_llm = get_original_llm()
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm_original_models/midm_original')

    tokenizer.pad_token = tokenizer.eos_token
    original_llm.generation_config.pad_token_id = tokenizer.pad_token_id  # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

    # read dataset
    dataset_df = convert_into_filled_df(f'{PROJECT_DIR_PATH}/ai_interview/dataset/all_train_and_test_data.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=original_llm, lora_rank=64)

    dataset_df['text'] = dataset_df.apply(
        lambda x: f"{x['input_data']} (발화 시작) ### 발화: {x['output_data']} (발화 종료) <|end_of_text|>",
        axis=1)
    dataset = generate_llm_trainable_dataset(dataset_df)
    preview_dataset(dataset, tokenizer)

    response_template = [28912, 28]  # '### 발화:'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = get_training_args(epochs)
    trainer = get_sft_trainer(dataset, collator, training_args, epochs)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    output_dir_path = f'{PROJECT_DIR_PATH}/ai_interview/models/midm_sft_final_fine_tuned_{epochs}epochs'
    trainer.save_model(output_dir_path)
