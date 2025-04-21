import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import peft
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl

import torch
import pandas as pd

from fine_tuning.inference import load_valid_user_prompts, run_inference


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
OUTPUT_DIR_PATH = f'{PROJECT_DIR_PATH}/llm/models/fine_tuned'

lora_llm = None
tokenizer = None
valid_user_prompts = load_valid_user_prompts()


class InferenceTestOnEpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_user_prompts

        print('=== INFERENCE TEST ===')

        for user_prompt in valid_user_prompts:
            llm_answer = run_inference(lora_llm, user_prompt, tokenizer)
            print(f'user prompt : {user_prompt}')
            print(f'llm answer : {llm_answer}')


# Original LLM (gemma-2 2b) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - original_llm (LLM) : Original Gemma-2 2B LLM

def get_original_llm():
    original_llm = AutoModelForCausalLM.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/original',
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16).cuda()

    return original_llm


# Original LLM (gemma-2 2b) 에 대한 Fine-Tuning 을 위한 Training Arguments 가져오기
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args():
    training_args = SFTConfig(
        learning_rate=0.0002,           # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=10,
        logging_steps=5,                # logging frequency
        gradient_checkpointing=False,
        output_dir=OUTPUT_DIR_PATH,
        save_total_limit=3,             # max checkpoint count to save
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size per device during validation
        report_to=None                  # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


# Original LLM (gemma-2 2b) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2025.04.21
# Last Update Date : 2025.04.22
# - 매 epoch 종료 시마다 Valid Data 로 Inference Test 를 하는 Callback 추가
# - lora_llm, tokenizer 를 global 변수로 수정

# Arguments:
# - dataset       (Dataset)       : LLM 학습 데이터셋
# - collator      (DataCollator)  : Data Collator
# - training_args (SFTConfig)     : Training Arguments

# Returns:
# - trainer (SFTTrainer) : SFT (Supervised Fine-Tuning) Trainer

def get_sft_trainer(dataset, collator, training_args):
    global lora_llm, tokenizer

    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,     # LLM tokenizer / renamed : tokenizer -> processing_class from trl 0.12.0
        args=training_args,
        data_collator=collator,
        callbacks=[InferenceTestOnEpochEndCallback()]
    )

    return trainer


# Original LLM (gemma-2 2b) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.04.21
# Last Update Date : 2025.04.22
# - lora_llm 을 global 변수로 수정

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (Gemma-2 2B)
# - lora_rank (int) : LoRA 적용 시의 Rank (권장: 64)

# Returns:
# - lora_llm (LLM) : LoRA 가 적용된 LLM

def get_lora_llm(llm, lora_rank):
    global lora_llm

    lora_config = LoraConfig(
        r=lora_rank,                    # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,              # Dropout for LoRA
        init_lora_weights="gaussian",   # LoRA weight initialization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    lora_llm = peft.get_peft_model(llm, lora_config)
    lora_llm.print_trainable_parameters()


# Original LLM (gemma-2 2b) 에 대한 LLM 이 직접 학습 가능한 데이터셋 가져오기
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame (from OhLoRA_fine_tuning.csv)
#                                   columns = ['data_type', 'input_data', 'output_data', 'output_message', 'memory']

# Returns:
# - dataset (Dataset) : LLM 학습 데이터셋

def generate_llm_trainable_dataset(dataset_df):
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'train'][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'valid'][['text']])

    print('\nLLM Trainable Dataset :')
    train_texts = dataset['train']['text']
    for i in range(10):
        print(f'train data {i} : {train_texts[i]}')
    print('\n')

    return dataset


# LLM (gemma-2 2b) Fine Tuning 실시
# Create Date : 2025.04.21
# Last Update Date : 2025.04.22
# - lora_llm 을 global 변수로 수정

# Arguments:
# - 없음

# Returns:
# - 2025_04_08_OhLoRA/llm/models/ohlora 에 Fine-Tuning 된 모델 저장

def fine_tune_model():
    global lora_llm, tokenizer

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    original_llm = get_original_llm()
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/original')

    # read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/OhLoRA_fine_tuning.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=original_llm, lora_rank=64)
    dataset_df['text'] = dataset_df.apply(lambda x: f"{x['input_data']} ### Answer: {x['output_data']}", axis=1)
    dataset = generate_llm_trainable_dataset(dataset_df)

    response_template = [43774, 10358, 235292]  # '### Answer :'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = get_training_args()
    trainer = get_sft_trainer(dataset, collator, training_args)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    trainer.save_model(OUTPUT_DIR_PATH)
