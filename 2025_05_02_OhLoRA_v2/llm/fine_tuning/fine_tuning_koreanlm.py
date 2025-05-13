import os
import os.path as osp
from typing import Union
import json

import peft
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl, DataCollatorForSeq2Seq

import torch
import pandas as pd

from fine_tuning.inference import load_valid_final_prompts, run_inference, run_inference_koreanlm
from fine_tuning.utils import get_instruction, koreanlm_tokenize


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

lora_llm = None
tokenizer = None
valid_final_prompts = None


# Modified Implementation from https://github.com/quantumaikr/KoreanLM/blob/main/utils.py (License: Apache 2.0)
# Original korean.json file (in this directory) from https://github.com/quantumaikr/KoreanLM/blob/main/templates/korean.json

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "korean"
        file_name = osp.join("templates", f"{PROJECT_DIR_PATH}/llm/fine_tuning/{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name, 'r', encoding='UTF8') as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


# Modified Implementation from https://github.com/quantumaikr/KoreanLM/blob/main/finetune-lora.py

def generate_and_tokenize_prompt(data_point, prompter, tokenizer, train_on_inputs=True):
    input_part = data_point['text'].split(' ### 답변: ')[0]
    output_part = data_point['text'].split(' ### 답변: ')[1]

    full_prompt = prompter.generate_prompt(get_instruction(), input_part, output_part)
    tokenized_full_prompt = koreanlm_tokenize(full_prompt, tokenizer, return_tensors=None)

    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(get_instruction(), input_part)
        tokenized_user_prompt = koreanlm_tokenize(user_prompt, tokenizer, return_tensors=None)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt


class InferenceTestOnEpochEndCallback(TrainerCallback):

    def __init__(self):
        super(InferenceTestOnEpochEndCallback, self).__init__()
        self.prompter = Prompter('korean')

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_final_prompts

        print('=== INFERENCE TEST ===')

        for final_input_prompt in valid_final_prompts:
            llm_answer, trial_count, output_token_cnt = run_inference_koreanlm(lora_llm,
                                                                               final_input_prompt,
                                                                               tokenizer,
                                                                               self.prompter)

            print(f'final input prompt : {final_input_prompt}')
            print(f'llm answer (trials: {trial_count}, output tkns: {output_token_cnt}) : {llm_answer}')


# Original LLM (KoreanLM 1.5B) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2025.05.12
# Last Update Date : 2025.05.12
# - KoreanLM-1.5B 를 Original KoreanLM-1.5B Fine-Tuning code 를 참고하여 변경

# Arguments:
# - 없음

# Returns:
# - original_llm (LLM) : Original KoreanLM 1.5B LLM

def get_original_llm():
    original_llm = AutoModelForCausalLM.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/koreanlm_original',
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.float16).cuda()

    return original_llm


# Original LLM (KoreanLM 1.5B) 에 대한 Fine-Tuning 을 위한 Training Arguments 가져오기
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - 업데이트된 학습 데이터셋 (OhLoRA_fine_tuning_v2.csv) 반영 및 총 4 개의 LLM 개별 학습
# - LLM output column 에 따라 서로 다른 training argument 적용

# Arguments:
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args(output_col):
    output_dir_path = f'{PROJECT_DIR_PATH}/llm/models/koreanlm_{output_col}_fine_tuned'
    num_train_epochs_dict = {'output_message': 80}
    num_train_epochs = num_train_epochs_dict[output_col]

    training_args = SFTConfig(
        learning_rate=0.0002,               # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=num_train_epochs,
        logging_steps=5,                    # logging frequency
        gradient_checkpointing=False,
        output_dir=output_dir_path,
        save_total_limit=3,                 # max checkpoint count to save
        per_device_train_batch_size=4,      # batch size per device during training
        per_device_eval_batch_size=1,       # batch size per device during validation
        fp16=True,
        report_to=None                      # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


# Original LLM (KoreanLM 1.5B) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2025.05.12
# Last Update Date : -

# Arguments:
# - dataset       (Dataset)      : LLM 학습 데이터셋
# - collator      (DataCollator) : Data Collator
# - training_args (SFTConfig)    : Training Arguments

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


# Original LLM (KoreanLM 1.5B) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2025.05.12
# Last Update Date : 2025.05.12
# - KoreanLM-1.5B 를 Original KoreanLM-1.5B Fine-Tuning code 를 참고하여 변경

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (KoreanLM 1.5B)
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
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    lora_llm = peft.get_peft_model(llm, lora_config)
    lora_llm.print_trainable_parameters()


# Original LLM (KoreanLM 1.5B) 에 대한 LLM 이 직접 학습 가능한 데이터셋 가져오기
# Create Date : 2025.05.12
# Last Update Date : 2025.05.12
# - KoreanLM-1.5B 를 Original KoreanLM-1.5B Fine-Tuning code 를 참고하여 변경

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame (from llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv)
#                                   columns = ['data_type', 'input_data', ...]
# - prompter   (Prompter)         : LLM 의 사용자 prompt & 답변의 입출력 형식을 나타내는 객체
# - tokenizer  (tokenizer)        : KoreanLM-1.5B 에 대한 tokenizer

# Returns:
# - dataset (Dataset) : LLM 학습 데이터셋

def generate_llm_trainable_dataset(dataset_df, prompter, tokenizer):
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'train'][['text']])
    dataset['train'] = dataset['train'].map(lambda x: generate_and_tokenize_prompt(x,
                                                                                   prompter=prompter,
                                                                                   tokenizer=tokenizer))

    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'valid'][['text']])
    dataset['valid'] = dataset['valid'].map(lambda x: generate_and_tokenize_prompt(x,
                                                                                   prompter=prompter,
                                                                                   tokenizer=tokenizer))

    print('\nLLM Trainable Dataset :')
    train_texts = dataset['train']['text']
    for i in range(10):
        print(f'train data {i} : {train_texts[i]}')
    print('\n')

    return dataset


# LLM (KoreanLM 1.5B) Fine-Tuning 실시
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - 업데이트된 학습 데이터셋 (OhLoRA_fine_tuning_v2.csv) 반영 및 총 4 개의 LLM 개별 학습

# Arguments:
# - output_col (str) : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name

# Returns:
# - 2025_05_02_OhLoRA_v2/llm/models/koreanlm_{output_col}_fine_tuned 에 Fine-Tuning 된 모델 저장

def fine_tune_model(output_col):
    global lora_llm, tokenizer, valid_final_prompts
    valid_final_prompts = load_valid_final_prompts(dataset_csv_path='llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv',
                                                   output_col=output_col)

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    # KoreanLM original model is from https://huggingface.co/quantumaikr/KoreanLM-1.5b/tree/main
    original_llm = get_original_llm()
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/koreanlm_original',
                                              padding_side='right')
    tokenizer.pad_token_id = 0  # unk

    # read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    prompter = Prompter('korean')

    if output_col == 'summary':
        dataset_df['text'] = dataset_df.apply(lambda x: f"{x['input_data'] + ' / ' + x['output_message']} ### 답변: {x[output_col]} <|endoftext|>",
                                              axis=1)
    else:
        dataset_df['text'] = dataset_df.apply(lambda x: f"{x['input_data']} ### 답변: {x[output_col]} <|endoftext|>",
                                              axis=1)

    dataset = generate_llm_trainable_dataset(dataset_df, prompter, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)

    get_lora_llm(llm=original_llm, lora_rank=128)
    training_args = get_training_args(output_col)
    trainer = get_sft_trainer(dataset, collator, training_args)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    output_dir_path = f'{PROJECT_DIR_PATH}/llm/models/koreanlm_{output_col}_fine_tuned'
    trainer.save_model(output_dir_path)
