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
OUTPUT_DIR_PATH = f'{PROJECT_DIR_PATH}/llm/models/koreanlm_fine_tuned'

lora_llm = None
tokenizer = None
valid_user_prompts = load_valid_user_prompts(dataset_csv_path='llm/fine_tuning_dataset/OhLoRA_fine_tuning_25042213.csv')


class InferenceTestOnEpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_user_prompts

        print('=== INFERENCE TEST ===')

        for user_prompt in valid_user_prompts:
            llm_answer, trial_count, output_token_cnt = run_inference(lora_llm,
                                                                      user_prompt,
                                                                      tokenizer,
                                                                      stop_token_list=[10234, 3082, 10904, 13],
                                                                      answer_start_mark=' (답변 시작)',
                                                                      remove_token_type_ids=True)

            print(f'user prompt : {user_prompt}')
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
# Last Update Date : 2025.05.12
# - KoreanLM-1.5B 를 Original KoreanLM-1.5B Fine-Tuning code 를 참고하여 변경

# Arguments:
# - 없음

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args():
    training_args = SFTConfig(
        learning_rate=0.0002,           # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=80,
        logging_steps=5,                # logging frequency
        gradient_checkpointing=False,
        output_dir=OUTPUT_DIR_PATH,
        save_total_limit=3,             # max checkpoint count to save
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size per device during validation
        fp16=True,
        report_to=None                  # to prevent wandb API key request at start of Fine-Tuning
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
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame (from OhLoRA_fine_tuning_25042213.csv)
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


# LLM (KoreanLM 1.5B) Fine-Tuning 실시
# Create Date : 2025.05.12
# Last Update Date : 2025.05.12
# - KoreanLM-1.5B 를 Original KoreanLM-1.5B Fine-Tuning code 를 참고하여 변경

# Arguments:
# - 없음

# Returns:
# - 2025_05_02_OhLoRA_v2/llm/models/koreanlm_fine_tuned 에 Fine-Tuning 된 모델 저장

def fine_tune_model():
    global lora_llm, tokenizer

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    # KoreanLM original model is from https://huggingface.co/quantumaikr/KoreanLM-1.5b/tree/main
    original_llm = get_original_llm()
    tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/koreanlm_original',
                                              padding_side='right')

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    original_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    # read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/fine_tuning_dataset/OhLoRA_fine_tuning_25042213.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=original_llm, lora_rank=128)

#    print(tokenizer.encode('### 답변:'))  # ... [20904, 14177, 3082, 30]
#    print(tokenizer.encode('(답변 시작) ### 답변:'))  # ... [12, 10234, 3082, 2211, 13, 225, 20904, 14177, 3082, 30]
#    print(tokenizer.encode('(답변 종료)'))  # ... [12, 10234, 3082, 10904, 13]

    dataset_df['text'] = dataset_df.apply(lambda x: f"{x['input_data']} (답변 시작) ### 답변: {x['output_data']} (답변 종료) <|endoftext|>",
                                          axis=1)
    dataset = generate_llm_trainable_dataset(dataset_df)

    response_template = [20904, 14177, 3082, 30]  # '### 답변 :'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = get_training_args()
    trainer = get_sft_trainer(dataset, collator, training_args)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    trainer.save_model(OUTPUT_DIR_PATH)
