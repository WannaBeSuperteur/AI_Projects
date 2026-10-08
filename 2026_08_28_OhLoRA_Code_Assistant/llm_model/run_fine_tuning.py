
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, Dataset

import peft
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, TrainerState, \
                         TrainerControl
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

from run_inference import LLMInferenceEngine
from utils import LLM_ORIGINAL_PATHS, TARGET_MODULES_DICT, STOP_TOKEN_LIST, ANSWER_START_MARK, ANSWER_END_MARK
from utils import add_train_log, add_inference_log


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'llm_model', 'train_log')
INFERENCE_LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'llm_model', 'inference_log')

os.makedirs(TRAIN_LOG_DIR_PATH, exist_ok=True)
os.makedirs(INFERENCE_LOG_DIR_PATH, exist_ok=True)


class OhLoRACustomCallback(TrainerCallback):

    def __init__(self, train_log_dict: dict, inference_log_dict: dict, llm_name: str, llm_path: str,
                 eval_dataset: list[str]):

        super(OhLoRACustomCallback, self).__init__()
        self.train_log_dict = train_log_dict
        self.inference_log_dict = inference_log_dict

        self.llm_name = llm_name
        self.llm_path = llm_path
        self.eval_dataset = eval_dataset

        self._init_inference_engine()

    def _init_inference_engine(self):
        self.inference_engine = LLMInferenceEngine(self.llm_path,
                                                   answer_start_mark=ANSWER_START_MARK,
                                                   answer_end_mark=ANSWER_END_MARK,
                                                   stop_token_list=STOP_TOKEN_LIST)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_log_df = pd.DataFrame(self.train_log_dict)
        train_log_df.to_csv(os.path.join(TRAIN_LOG_DIR_PATH, f'{self.llm_name}.csv'))

        for final_input_prompt in self.eval_dataset:
            start_at = time.time()

            self.inference_engine.load_llm()
            inference_result = self.inference_engine.run_inference(final_input_prompt)
            self.inference_engine.unload_llm()

            llm_answer, trial_cnt, output_token_cnt = (
                inference_result['llm_answer'], inference_result['trial_cnt'], inference_result['output_token_cnt'])

            llm_answer = llm_answer[:-len(ANSWER_END_MARK) + 1]
            elapsed_time = time.time() - start_at

            print(f'final input prompt : {final_input_prompt}')
            print(f'llm answer (trials: {trial_cnt}, output tkns: {output_token_cnt}) : {llm_answer}')

            inference_result = {'epoch': state.epoch, 'elapsed_time': elapsed_time, 'prompt': final_input_prompt,
                                'llm_answer': llm_answer, 'trial_cnt': trial_cnt, 'output_tkn_cnt': output_token_cnt}
            add_inference_log(inference_result, self.inference_log_dict)

        inference_log_df = pd.DataFrame(self.inference_log_dict)
        inference_log_df.to_csv(os.path.join(INFERENCE_LOG_DIR_PATH, f'{self.llm_name}.csv'))

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            add_train_log(state, self.train_log_dict)
        except Exception as e:
            print(f'logging failed : {e}')


class LLMTrainer():
    def __init__(self, original_path: str, save_path: str):
        self.original_path = original_path
        self.save_path = save_path

        self.llm_name = original_path.split('/')[-1].lower()
        self.target_modules = TARGET_MODULES_DICT.get(self.llm_name) or TARGET_MODULES_DICT['default']

        self.original_llm = self._get_original_llm()
        self.tokenizer = AutoTokenizer.from_pretrained(self.original_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_log_dict = {'epoch': [],
                               'time': [],
                               'loss': [],
                               'grad_norm': [],
                               'learning_rate': [],
                               'mean_token_accuracy': []}
        self.inference_log_dict = {'epoch': [],
                                   'elapsed_time (s)': [],
                                   'prompt': [],
                                   'llm_answer': [],
                                   'trial_cnt': [],
                                   'output_tkn_cnt': []}

    def _generate_llm_trainable_dataset(self, dataset_df):
        dataset = DatasetDict()
        dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['split'] == 'train'][['text']])
        dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['split'] == 'valid'][['text']])

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
            report_to="none"                     # to prevent wandb API key request at start of Fine-Tuning
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
            data_collator=collator,
            callbacks=[OhLoRACustomCallback(self.train_log_dict,
                                            self.inference_log_dict,
                                            self.llm_name,
                                            self.save_path,
                                            list(dataset['valid']))]
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

        dataset_df = pd.read_csv(os.path.join(PROJECT_DIR_PATH, "ai_dataset", "llm_dataset", "llm_dataset.csv"))
        dataset_df = dataset_df.sample(frac=1, random_state=2026)  # shuffle
        dataset_df['split'] = np.where(np.arange(len(dataset_df)) < len(dataset_df) * 0.8, 'train', 'valid')

        # prepare Fine-Tuning
        self._get_lora_llm(llm=self.original_llm)

        dataset_df['text'] = dataset_df.apply(
            lambda x: f"{x['input']} (답변 시작) ### 답변: {x['output']}{ANSWER_END_MARK}",
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
