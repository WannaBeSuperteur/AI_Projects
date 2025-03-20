import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from datasets import DatasetDict, Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_values import PROMPT_PREFIX, PROMPT_SUFFIX
from sklearn.model_selection import train_test_split
from common import compute_output_score, add_text_column_for_llm
from peft import LoraConfig, get_peft_model

import pandas as pd


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# to prevent "RuntimeError: chunk expects at least a 1-dimensional tensor"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# SFT 실시
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - df_train (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Train Data
#                                 columns: ['input_data', 'output_data']
# - df_valid (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Valid Data
#                                 columns: ['input_data', 'output_data']

# Returns:
# - llm (LLM) : SFT 로 Fine-tuning 된 LLM

def run_fine_tuning(df_train, df_valid):

    print(f'Train DataFrame:\n{df_train}')
    print(f'Valid DataFrame:\n{df_valid}')

    model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"
    output_dir = "sft_model"

    original_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                        torch_dtype=torch.float16).cuda()
    original_llm.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'right'

    lora_config = LoraConfig(
        r=16,  # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,  # Dropout for LoRA
        init_lora_weights="gaussian",  # LoRA weight initialization
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
    )
    lora_llm = get_peft_model(original_llm, lora_config)
    lora_llm.print_trainable_parameters()

    training_args = SFTConfig(
        learning_rate=0.0002,  # lower learning rate is recommended for fine tuning
        num_train_epochs=2,
        logging_steps=5,  # logging frequency
        gradient_checkpointing=True,
        output_dir=output_dir,
        save_total_limit=3,  # max checkpoint count to save
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4  # batch size per device during validation
    )

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(df_train)
    dataset['valid'] = Dataset.from_pandas(df_valid)

    response_template = '### Answer:'
    response_template = tokenizer.encode(f"\n{response_template}", add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        dataset_text_field='text',
        tokenizer=tokenizer,
        max_seq_length=2048,  # 한번에 입력 가능한 최대 token 개수 (클수록 GPU 메모리 사용량 증가)
        args=training_args,
        data_collator=collator,
        compute_metrics=compute_output_score
    )

    trainer.train()
    trainer.save_model(output_dir)

    checkpoint_output_dir = os.path.join(output_dir, 'deepseek_checkpoint')
    trainer.model.save_pretrained(checkpoint_output_dir)
    tokenizer.save_pretrained(checkpoint_output_dir)

    return lora_llm


# SFT 테스트를 위한 모델 로딩
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - llm (LLM) : SFT 로 Fine-tuning 된 LLM

def load_sft_llm():
    raise NotImplementedError


# SFT 로 Fine-Tuning 된 LLM 을 테스트
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - llm              (LLM)       : SFT 로 Fine-tuning 된 LLM
# - llm_prompts      (list(str)) : 해당 LLM 에 전달할 User Prompt (Prompt Engineering 을 위해 추가한 부분 제외)
# - llm_dest_outputs (list(str)) : 해당 LLM 의 목표 output 답변

# Returns:
# - llm_answers (list(str)) : 해당 LLM 의 답변
# - score       (float)     : 해당 LLM 의 성능 score

def test_sft_llm(llm, llm_prompts, llm_dest_outputs):
    raise NotImplementedError


if __name__ == '__main__':

    # check cuda is available
    assert torch.cuda.is_available(), "CUDA MUST BE AVAILABLE"
    print(f'cuda is available with device {torch.cuda.get_device_name()}')

    sft_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/sft_dataset_llm.csv'
    df = pd.read_csv(sft_dataset_path)

    add_text_column_for_llm(df)  # LLM 이 학습할 수 있도록 text column 추가

    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=2025)

    # LLM Fine-tuning
    llm = run_fine_tuning(df_train, df_valid)

    # LLM 테스트
    llm_for_test = load_sft_llm()
    llm_prompts = df_valid['input_data'].tolist()
    llm_dest_outputs = df_valid['output_data'].tolist()

    llm_answer, score = test_sft_llm(llm_for_test, llm_prompts, llm_dest_outputs)

    print(f'\nLLM Score :\n{score}')
