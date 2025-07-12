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
    from llm_fine_tuning.inference import run_inference_kanana
    from llm_fine_tuning.utils import load_valid_final_prompts, preview_dataset, add_train_log, add_inference_log, \
                                  get_answer_start_mark
    from llm_fine_tuning.common import convert_into_filled_df

except:
    from ai_quiz.llm_fine_tuning.inference import run_inference_kanana
    from ai_quiz.llm_fine_tuning.utils import load_valid_final_prompts, preview_dataset, add_train_log, \
        add_inference_log, get_answer_start_mark
    from ai_quiz.llm_fine_tuning.common import convert_into_filled_df


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

lora_llm = None
tokenizer = None
valid_final_prompts = None

train_log_dict = {'epoch': [], 'time': [], 'loss': [], 'grad_norm': [], 'learning_rate': [], 'mean_token_accuracy': []}
inference_log_dict = {'epoch': [], 'elapsed_time (s)': [], 'prompt': [], 'llm_answer': [],
                      'trial_cnt': [], 'output_tkn_cnt': []}

log_dir_path = f'{PROJECT_DIR_PATH}/ai_quiz/llm_fine_tuning/logs'
os.makedirs(log_dir_path, exist_ok=True)


def get_stop_token_list():
    return [34983, 102546, 99458, 64356]  # (해설 종료)


class OhLoRACustomCallback(TrainerCallback):

    def __init__(self, instruct_version):
        super(OhLoRACustomCallback, self).__init__()
        self.instruct_version = instruct_version

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_final_prompts

        kanana_llm_name = 'kananai' if self.instruct_version else 'kanana'
        train_log_df = pd.DataFrame(train_log_dict)
        train_log_df.to_csv(f'{log_dir_path}/{kanana_llm_name}_sft_final_train_log.csv')

        print('=== INFERENCE TEST ===')

        for final_input_prompt in valid_final_prompts:
            start_at = time.time()
            stop_token_list = get_stop_token_list()
            answer_start_mark = get_answer_start_mark()

            llm_answer, trial_count, output_token_cnt = run_inference_kanana(lora_llm,
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
        inference_log_df.to_csv(f'{log_dir_path}/{kanana_llm_name}_sft_final_inference_log_dict.csv')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            add_train_log(state, train_log_dict)
        except Exception as e:
            print(f'logging failed : {e}')


# Original LLM (Kanana-1.5 2.1B) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - kanana_llm_name (str) : 'kananai' for Kanana-1.5-2.1B instruct, 'kanana' for Kanana-1.5-2.1B base

# Returns:
# - original_llm (LLM) : Original Kanana-1.5 2.1B LLM

def get_original_llm(kanana_llm_name):
    original_llm_path = f'{PROJECT_DIR_PATH}/llm_original_models/{kanana_llm_name}_original'

    original_llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=original_llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).cuda()

    print(f'Original LLM load successful : {original_llm_path}')
    return original_llm


# Original LLM (Kanana-1.5 2.1B) 에 대한 Fine-Tuning 을 위한 Training Arguments 가져오기
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - kanana_llm_name (str) : 'kananai' for Kanana-1.5-2.1B instruct, 'kanana' for Kanana-1.5-2.1B base

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args(kanana_llm_name):
    output_dir_path = f'{PROJECT_DIR_PATH}/ai_quiz/models/{kanana_llm_name}_sft_final_fine_tuned'
    num_train_epochs = 5

    training_args = SFTConfig(
        learning_rate=0.0003,               # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=num_train_epochs,
        logging_steps=5,                    # logging frequency
        gradient_checkpointing=False,
        output_dir=output_dir_path,
        save_total_limit=3,                 # max checkpoint count to save
        per_device_train_batch_size=1,      # batch size per device during training
        per_device_eval_batch_size=1,       # batch size per device during validation
        report_to=None                      # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


# Original LLM (Kanana-1.5 2.1B) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - dataset          (Dataset)      : LLM 학습 데이터셋
# - collator         (DataCollator) : Data Collator
# - training_args    (SFTConfig)    : Training Arguments
# - instruct_version (bool)         : True for Kanana-1.5-2.1B instruct, False for Kanana-1.5-2.1B base

# Returns:
# - trainer (SFTTrainer) : SFT (Supervised Fine-Tuning) Trainer

def get_sft_trainer(dataset, collator, training_args, instruct_version):
    global lora_llm, tokenizer

    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,     # LLM tokenizer / renamed : tokenizer -> processing_class from trl 0.12.0
        args=training_args,
        data_collator=collator,
        callbacks=[OhLoRACustomCallback(instruct_version)]
    )

    return trainer


# Original LLM (Kanana-1.5 2.1B) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.07.12
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
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame (from ai_quiz/dataset/all_train_and_test_data.csv)
#                                   columns = ['data_type', 'quiz', 'keywords', ...]

# Returns:
# - dataset (Dataset) : LLM 학습 데이터셋

def generate_llm_trainable_dataset(dataset_df):
    global tokenizer

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'].str.startswith('train')][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'].str.startswith('valid')][['text']])
    preview_dataset(dataset, tokenizer)

    return dataset


# LLM (Kanana-1.5 2.1B) Fine-Tuning 실시
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - instruct_version (bool) : True for Kanana-1.5-2.1B instruct, False for Kanana-1.5-2.1B base

# Returns:
# - ai_qna/models/{kanana|kananai}_sft_final_fine_tuned 에 Fine-Tuning 된 모델 저장

def fine_tune_model(instruct_version):
    global lora_llm, tokenizer, valid_final_prompts
    valid_final_prompts = load_valid_final_prompts()
    kanana_llm_name = 'kananai' if instruct_version else 'kanana'

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    # Kanana-1.5 2.1B original model is from https://huggingface.co/kakaocorp/kanana-1.5-2.1b-base (base)
    #                                     or https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505 (instruct)
    original_llm = get_original_llm(kanana_llm_name)
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm_original_models/{kanana_llm_name}_original')

    tokenizer.pad_token = tokenizer.eos_token
    original_llm.generation_config.pad_token_id = tokenizer.pad_token_id  # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

    # read dataset
    dataset_df = convert_into_filled_df(f'{PROJECT_DIR_PATH}/ai_quiz/dataset/all_train_and_test_data.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=original_llm, lora_rank=64)

    dataset_df['text'] = dataset_df.apply(
        lambda x: f"{x['input_data']} (해설 시작) ### 해설: {x['explanation']} (해설 종료) <|eot_id|>",
        axis=1)
    dataset = generate_llm_trainable_dataset(dataset_df)
    preview_dataset(dataset, tokenizer)

    response_template = [61816, 102546]  # '### 해설 :'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = get_training_args(kanana_llm_name)
    trainer = get_sft_trainer(dataset, collator, training_args, instruct_version)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    output_dir_path = f'{PROJECT_DIR_PATH}/ai_quiz/models/{kanana_llm_name}_sft_final_fine_tuned'
    trainer.save_model(output_dir_path)
