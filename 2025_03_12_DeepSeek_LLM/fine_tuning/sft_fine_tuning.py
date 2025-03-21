import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from datasets import DatasetDict, Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.model_selection import train_test_split
from common import compute_output_score, add_text_column_for_llm
from draw_diagram.draw_diagram import generate_diagram_from_lines
from peft import LoraConfig, get_peft_model

import pandas as pd
import numpy as np
import time


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# to prevent "RuntimeError: chunk expects at least a 1-dimensional tensor"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = '1'


# SFT 실시
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - df_train (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Train Data
#                                 columns: ['input_data', 'output_data', 'dest_shape_info']
# - df_valid (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Valid Data
#                                 columns: ['input_data', 'output_data', 'dest_shape_info']

# Returns:
# - lora_llm  (LLM)       : SFT 로 Fine-tuning 된 LLM
# - tokenizer (tokenizer) : 해당 LLM 에 대한 tokenizer

def run_fine_tuning(df_train, df_valid):

    print(f'Train DataFrame:\n{df_train}')
    print(f'Valid DataFrame:\n{df_valid}')

    model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"
    output_dir = "sft_model"

    original_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                        torch_dtype=torch.float16).cuda()
    original_llm.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_path, eos_token='<eos>')
    tokenizer.pad_token = tokenizer.eos_token
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
        num_train_epochs=4,
        logging_steps=1,  # logging frequency
        gradient_checkpointing=True,
        output_dir=output_dir,
        save_total_limit=3,  # max checkpoint count to save
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1  # batch size per device during validation
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
        max_seq_length=1536,  # 한번에 입력 가능한 최대 token 개수 (클수록 GPU 메모리 사용량 증가)
        args=training_args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir)

    checkpoint_output_dir = os.path.join(output_dir, 'deepseek_checkpoint')
    trainer.model.save_pretrained(checkpoint_output_dir)
    tokenizer.save_pretrained(checkpoint_output_dir)

    return lora_llm, tokenizer


# SFT 테스트를 위한 모델 로딩
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - llm       (LLM)       : SFT 로 Fine-tuning 된 LLM (없으면 None)
# - tokenizer (tokenizer) : 해당 LLM 에 대한 tokenizer (LLM 이 없으면 None)

def load_sft_llm():
    print('loading LLM ...')

    try:
        model = AutoModelForCausalLM.from_pretrained("sft_model").cuda()
        tokenizer = AutoTokenizer.from_pretrained("sft_model", eos_token='<eos>')
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    except Exception as e:
        print(f'loading LLM failed : {e}')
        return None, None


# SFT 로 Fine-Tuning 된 LLM 을 테스트
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - llm              (LLM)        : SFT 로 Fine-tuning 된 LLM
# - tokenizer        (tokenizer)  : 해당 LLM 의 tokenizer
# - shape_infos      (list(dict)) : 해당 LLM 이 valid/test dataset 의 각 prompt 에 대해 생성해야 할 도형들에 대한 정보
# - llm_prompts      (list(str))  : 해당 LLM 에 전달할 User Prompt (Prompt Engineering 을 위해 추가한 부분 제외)
# - llm_dest_outputs (list(str))  : 해당 LLM 의 목표 output 답변

# Returns:
# - llm_answers (list(str)) : 해당 LLM 의 답변
# - final_score (float)     : 해당 LLM 의 성능 score
#
# - log/log_llm_test_result.csv 에 해당 LLM 테스트 기록 저장
# - 모델의 답변을 이용하여, sft_model_diagrams/diagram_{k}.png 다이어그램 파일 생성

def test_sft_llm(llm, tokenizer, shape_infos, llm_prompts, llm_dest_outputs):
    result_dict = {'prompt': [], 'answer': [], 'dest_output': [], 'time': [], 'score': []}

    for idx, (prompt, shape_info, dest_output) in enumerate(zip(llm_prompts, shape_infos, llm_dest_outputs)):
        inputs = tokenizer(f'### Question: {prompt}\n ### Answer: ',  return_tensors='pt').to(llm.device)

        with torch.no_grad():
            if idx == 0:
                print('llm output generating ...')

            start = time.time()
            outputs = llm.generate(**inputs, max_length=1536, do_sample=True)
            generate_time = time.time() - start
            llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace('<|EOT|>', '')
            llm_answer = llm_answer.split('### Answer: ')[1]  # prompt 부분을 제외한 answer 만 표시

        # llm answer 를 이용하여 Diagram 생성 및 저장
        try:
            llm_answer_lines = llm_answer.split('\n')
            os.makedirs(f'{PROJECT_DIR_PATH}/fine_tuning/sft_model_diagrams', exist_ok=True)
            diagram_save_path = f'{PROJECT_DIR_PATH}/fine_tuning/sft_model_diagrams/diagram_{idx:06d}.png'
            generate_diagram_from_lines(llm_answer_lines, diagram_save_path)

        except Exception as e:
            print(f'SFT diagram generation failed: {e}')

        score = compute_output_score(shape_info=shape_info, output_data=llm_answer)

        result_dict['prompt'].append(prompt)
        result_dict['answer'].append(llm_answer)
        result_dict['dest_output'].append(dest_output)
        result_dict['time'].append(generate_time)
        result_dict['score'].append(score)

        pd.DataFrame(result_dict).to_csv(f'{PROJECT_DIR_PATH}/fine_tuning/log/log_llm_test_result.csv')

    llm_answers = result_dict['answer']
    final_score = np.mean(result_dict['score'])

    return llm_answers, final_score


if __name__ == '__main__':

    # check cuda is available
    assert torch.cuda.is_available(), "CUDA MUST BE AVAILABLE"
    print(f'cuda is available with device {torch.cuda.get_device_name()}')

    sft_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/sft_dataset_llm.csv'
    df = pd.read_csv(sft_dataset_path)

    add_text_column_for_llm(df)  # LLM 이 학습할 수 있도록 text column 추가

    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=2025)

    # LLM Fine-tuning
    llm, tokenizer = load_sft_llm()

    if llm is None or tokenizer is None:
        print('LLM load failed, fine tuning ...')
        llm, tokenizer = run_fine_tuning(df_train, df_valid)
    else:
        print('LLM load successful!')

    # LLM 테스트
    print('LLM test start')

    llm_prompts = df_valid['input_data'].tolist()
    llm_dest_outputs = df_valid['output_data'].tolist()
    shape_infos = df_valid['dest_shape_info'].tolist()

    llm_answer, final_score = test_sft_llm(llm, tokenizer, shape_infos, llm_prompts, llm_dest_outputs)
    print(f'\nLLM FINAL Score :\n{final_score}')
