import os
import torch
import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOTrainer, ORPOConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# ORPO Fine-Tuning 을 위해 Pandas DataFrame 을 ORPO 형식 {"prompt": [...], "chosen": [...], "rejected": [...]} 으로 변환
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - df_train (Pandas DataFrame) : ORPO 형식 Train Data 의 DataFrame
#                                 columns: ['prompt', 'chosen', 'rejected']
# - df_valid (Pandas DataFrame) : ORPO 형식 Valid Data 의 DataFrame
#                                 columns: ['prompt', 'chosen', 'rejected']

# Returns:
# - orpo_train_dict (dict(list)) : Train Data 가 ORPO 로 직접 학습 가능한 데이터 형식으로 변환된 데이터셋
#                                  형식: {"prompt": [...], "chosen": [...], "rejected": [...]}
# - orpo_valid_dict (dict(list)) : Valid Data 가 ORPO 로 직접 학습 가능한 데이터 형식으로 변환된 데이터셋
#                                  형식: {"prompt": [...], "chosen": [...], "rejected": [...]}

def convert_df_to_orpo_format(df_train, df_valid):
    orpo_train_dict = {
        'prompt': df_train['prompt'].apply(lambda x: f"### Question: {x}\n ### Answer: ").tolist(),
        'chosen': df_train['chosen'].apply(lambda x: f"{x}<eos>").tolist(),
        'rejected': df_train['rejected'].apply(lambda x: f"{x}<eos>").tolist()
    }
    orpo_valid_dict = {
        'prompt': df_valid['prompt'].apply(lambda x: f"### Question: {x}\n ### Answer: ").tolist(),
        'chosen': df_valid['chosen'].apply(lambda x: f"{x}<eos>").tolist(),
        'rejected': df_valid['rejected'].apply(lambda x: f"{x}<eos>").tolist()
    }

    return orpo_train_dict, orpo_valid_dict


# ORPO Fine-Tuning 실시
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - df_train (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Train Data
#                                 columns: ['input_data', 'output_data', 'dest_shape_info']
# - df_valid (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Valid Data
#                                 columns: ['input_data', 'output_data', 'dest_shape_info']

# Returns:
# - lora_llm  (LLM)       : ORPO 로 Fine-tuning 된 LLM
# - tokenizer (tokenizer) : 해당 LLM 에 대한 tokenizer

def run_fine_tuning(df_train, df_valid):

    # Dataset 변환
    orpo_train, orpo_valid = convert_df_to_orpo_format(df_train, df_valid)

    orpo_train_dataset = Dataset.from_dict(orpo_train)
    orpo_valid_dataset = Dataset.from_dict(orpo_valid)

    # SFT 학습된 모델 및 tokenizer 불러오기
    model_path = f"{PROJECT_DIR_PATH}/sft_model"
    output_dir = "orpo_model"

    sft_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                   torch_dtype=torch.float16).cuda()
    sft_llm = prepare_model_for_kbit_training(sft_llm)
    sft_llm.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_path, eos_token='<eos>')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Training Configurations (LoRA, ORPO, ...)
    lora_config = LoraConfig(
        r=16,                           # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,              # Dropout for LoRA
        init_lora_weights="gaussian",   # LoRA weight initialization
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
    )
    lora_llm = get_peft_model(sft_llm, lora_config)
    lora_llm.print_trainable_parameters()

    training_args = ORPOConfig(
        learning_rate=1e-5,                # lower learning rate is recommended for fine tuning
        num_train_epochs=4,
        logging_steps=1,                   # logging frequency
        optim="adamw_8bit",                # memory-efficient AdamW optimizer
        gradient_checkpointing=True,
        output_dir=output_dir,
        save_total_limit=3,                # max checkpoint count to save
        per_device_train_batch_size=1,     # batch size per device during training
        per_device_eval_batch_size=1,      # batch size per device during validation
        max_length=1536,                   # max length of (prompt + LLM answer)
        max_prompt_length=512,             # max length of (ONLY prompt)
        remove_unused_columns=False
    )

    dataset = DatasetDict()
    dataset['train'] = orpo_train_dataset
    dataset['valid'] = orpo_valid_dataset

    print(dataset['train'])
    print(dataset['valid'])

    trainer = ORPOTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()
    trainer.save_model(output_dir)

    checkpoint_output_dir = os.path.join(output_dir, 'deepseek_checkpoint')
    trainer.model.save_pretrained(checkpoint_output_dir)
    tokenizer.save_pretrained(checkpoint_output_dir)

    return lora_llm, tokenizer


# ORPO 테스트를 위한 모델 로딩
# Create Date : 2025.03.23
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - llm (LLM) : SFT + ORPO 로 Fine-tuning 된 LLM

def load_orpo_llm():
    print('loading LLM ...')

    try:
        model = AutoModelForCausalLM.from_pretrained(f"{PROJECT_DIR_PATH}/orpo_model").cuda()
        model = torch.compile(model)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(f"{PROJECT_DIR_PATH}/orpo_model", eos_token='<eos>')
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    except Exception as e:
        print(f'loading LLM failed : {e}')
        return None, None


# SFT + ORPO 로 Fine-Tuning 된 LLM 을 테스트
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - llm              (LLM)       : SFT + ORPO 로 Fine-tuning 된 LLM
# - llm_prompts      (list(str)) : 해당 LLM 에 전달할 User Prompt (Prompt Engineering 을 위해 추가한 부분 제외)
# - llm_dest_outputs (list(str)) : 해당 LLM 의 목표 output 답변

# Returns:
# - llm_answers (list(str)) : 해당 LLM 의 답변
# - score       (float)     : 해당 LLM 의 성능 score

def test_orpo_llm(llm, llm_prompts, llm_dest_outputs):
    raise NotImplementedError


if __name__ == '__main__':

    # check cuda is available
    assert torch.cuda.is_available(), "CUDA MUST BE AVAILABLE"
    print(f'cuda is available with device {torch.cuda.get_device_name()}')

    orpo_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/orpo_dataset_llm.csv'
    df = pd.read_csv(orpo_dataset_path, index_col=0)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=2025)

    # LLM Fine-tuning
    llm, tokenizer = load_orpo_llm()

    if llm is None or tokenizer is None:
        print('LLM load failed, fine tuning ...')
        llm, tokenizer = run_fine_tuning(df_train, df_valid)
    else:
        print('LLM load successful!')

    # LLM 테스트
    print('LLM test start')

    llm_prompts = df_valid['input_data'].tolist()
    llm_dest_outputs = df_valid['output_data'].tolist()

    llm_answer, score = test_orpo_llm(llm, llm_prompts, llm_dest_outputs)

    print(f'\nLLM Score :\n{score}')
