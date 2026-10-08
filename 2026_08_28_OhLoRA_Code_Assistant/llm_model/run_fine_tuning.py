
import os

import pandas as pd
import torch
from datasets import DatasetDict, Dataset

import peft
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

from utils import LLM_ORIGINAL_PATHS, TARGET_MODULES_DICT


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


class LLMTrainer():
    def __init__(self, original_path: str, save_path: str):
        self.original_path = original_path
        self.save_path = save_path

        llm_name = original_path.split('/')[-1]
        self.target_modules = 'default' if llm_name in TARGET_MODULES_DICT else TARGET_MODULES_DICT[llm_name]

        self.original_llm = self._get_original_llm()
        self.tokenizer = AutoTokenizer.from_pretrained(self.original_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate_llm_trainable_dataset(self, dataset_df):
        dataset = DatasetDict()
        dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'train'][['text']])
        dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'valid'][['text']])

        print('\nLLM Trainable Dataset :')
        train_texts = dataset['train']['text']
        for i in range(10):
            print(f'train data {i} : {train_texts[i]}')
        print('\n')

        return dataset

    def _preview_dataset(self, dataset, print_encoded_tokens=False):
        print('\n=== DATASET PREVIEW ===')
        print(f"dataset size: [train: {len(dataset['train']['text'])}, valid: {len(dataset['valid']['text'])}]")

        for i in range(10):
            print(f"\ntrain data {i}: {dataset['train']['text'][i]}")
            if print_encoded_tokens:
                print(f"train data {i} tokenized: {self.tokenizer.encode(dataset['train']['text'][i])}")

            print(f"valid data {i} : {dataset['valid']['text'][i].split('###')[0]}")
            if print_encoded_tokens:
                print(f"valid data {i} tokenized: {self.tokenizer.encode(dataset['valid']['text'][i].split('###')[0])}")

        print('')

    def _get_training_args(self, num_train_epochs):
        training_args = SFTConfig(
            learning_rate=0.0003,                # lower learning rate is recommended for Fine-Tuning
            num_train_epochs=num_train_epochs,
            logging_steps=5,                     # logging frequency
            gradient_checkpointing=False,
            output_dir=self.save_path,
            save_total_limit=3,                  # max checkpoint count to save
            per_device_train_batch_size=2,       # batch size per device during training
            per_device_eval_batch_size=1,        # batch size per device during validation
            report_to=None                       # to prevent wandb API key request at start of Fine-Tuning
        )

        return training_args

    def _get_original_llm(self):
        original_llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.original_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()

        return original_llm

    def _get_sft_trainer(self, dataset, collator, training_args):
        self.sft_trainer = SFTTrainer(
            self.lora_llm,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            processing_class=self.tokenizer,
            args=training_args,
            data_collator=collator
        )

    def _get_lora_llm(self, llm):
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,             # Dropout for LoRA
            init_lora_weights="gaussian",  # LoRA weight initialization
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )

        self.lora_llm = peft.get_peft_model(llm, lora_config)
        self.lora_llm.print_trainable_parameters()

    def _fine_tune_llm(self):
        print(f'{self.original_path} LLM Fine Tuning start.')

        # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
        self.original_llm.generation_config.pad_token_id = self.tokenizer.pad_token_id

        dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/train_data.csv')
        dataset_df = dataset_df.sample(frac=1)  # shuffle

        # prepare Fine-Tuning
        self._get_lora_llm(llm=self.original_llm)

        dataset_df['text'] = dataset_df.apply(
            lambda x: f"{x['input_data']} (답변 시작) ### 답변: {x['output_message']} (답변 종료) <|end_of_text|>",
            axis=1)
        dataset = self._generate_llm_trainable_dataset(dataset_df)
        self._preview_dataset(dataset)

        response_template = [8, 17010, 111964, 25]  # '### 답변 :'

        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        training_args = self._get_training_args(num_train_epochs=5)
        self._get_sft_trainer(dataset, collator, training_args)

        # run Fine-Tuning
        self.sft_trainer.train()

    def run(self):
        """Train LLM."""

        self._fine_tune_llm()

    def save_llm(self):
        """Save LLM into save path. (Full LLM)"""

        self.sft_trainer.save_model(self.save_path)


def train_and_save_llm(original_path: str, save_path: str):
    llm_trainer = LLMTrainer(original_path, save_path)
    llm_trainer.run()
    llm_trainer.save_llm()


if __name__ == '__main__':
    for original_path in LLM_ORIGINAL_PATHS:
        save_path = os.path.join(PROJECT_DIR_PATH, original_path.split('/')[-1].lower())
        save_path = str(save_path)
        train_and_save_llm(original_path, save_path)
