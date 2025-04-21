import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM
import torch

import peft
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict, Dataset


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
OUTPUT_DIR_PATH = f'{PROJECT_DIR_PATH}/llm/models/fine_tuned'


# Original LLM (gemma-2 2b) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - original_llm (LLM) : Original Gemma-2 2B LLM

def get_llm():
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
# Last Update Date : -

# Arguments:
# - lora_llm      (LLM)           : LoRA 가 적용된 LLM
# - dataset       (Dataset)       : LLM 학습 데이터셋
# - tokenizer     (AutoTokenizer) : LLM 의 Tokenizer
# - collator      (DataCollator)  : Data Collator
# - training_args (SFTConfig)     : Training Arguments

# Returns:
# - trainer (SFTTrainer) : SFT (Supervised Fine-Tuning) Trainer

def get_sft_trainer(lora_llm, dataset, tokenizer, collator, training_args):
    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,     # LLM tokenizer / renamed : tokenizer -> processing_class from trl 0.12.0
        args=training_args,
        data_collator=collator
    )

    return trainer


# Original LLM (gemma-2 2b) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (Gemma-2 2B)
# - lora_rank (int) : LoRA 적용 시의 Rank (권장: 64)

# Returns:
# - lora_llm (LLM) : LoRA 가 적용된 LLM

def get_lora_llm(llm, lora_rank):
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

    return lora_llm


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

    return dataset


# LLM (gemma-2 2b) Fine Tuning 실시
# Create Date : 2025.04.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - 2025_04_08_OhLoRA/llm/models/ohlora 에 Fine-Tuning 된 모델 저장

def fine_tune_model():
    raise NotImplementedError
