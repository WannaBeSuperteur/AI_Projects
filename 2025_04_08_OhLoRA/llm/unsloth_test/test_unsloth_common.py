import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from unsloth import FastLanguageModel
except:
    pass

import peft
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict, Dataset

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


llm_path = 'unsloth/gemma-2-2b-it'
TEST_PROMPT_COUNT = 25
START_PROMPT_IDX = 81

output_dir_path = f'{PROJECT_DIR_PATH}/llm/unsloth_test/model_output_without_unsloth'
output_dir_path_with_unsloth = f'{PROJECT_DIR_PATH}/llm/unsloth_test/model_output_with_unsloth'


def get_training_args(with_unsloth):
    grad_ckpting = 'unsloth' if with_unsloth else False

    training_args = SFTConfig(
        learning_rate=0.0002,                 # lower learning rate is recommended for fine tuning
        num_train_epochs=3,
        logging_steps=1,                      # logging frequency
        gradient_checkpointing=grad_ckpting,
        output_dir=output_dir_path,
        save_total_limit=3,                   # max checkpoint count to save
        per_device_train_batch_size=4,        # batch size per device during training
        per_device_eval_batch_size=1,         # batch size per device during validation
        report_to=None                        # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


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


def get_lora_llm(llm, lora_rank, with_unsloth=False):
    lora_config = LoraConfig(
        r=lora_rank,                    # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,              # Dropout for LoRA
        init_lora_weights="gaussian",   # LoRA weight initialization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    if with_unsloth:
        lora_llm = FastLanguageModel.get_peft_model(llm, lora_config)
    else:
        lora_llm = peft.get_peft_model(llm, lora_config)

    lora_llm.print_trainable_parameters()

    return lora_llm


def generate_llm_trainable_dataset(dataset_df):
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'train'][['text']][120:140])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'valid'][['text']][5:10])

    return dataset
