import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

import torch
import pandas as pd
import time

from test_unsloth_common import llm_path, output_dir_path_with_unsloth
from test_unsloth_common import TEST_PROMPT_COUNT, START_PROMPT_IDX
from test_unsloth_common import get_lora_llm, generate_llm_trainable_dataset, get_sft_trainer, get_training_args

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device for testing LLM Fine-Tuning with Unsloth : {device}')


def get_llm():
    llm = FastLanguageModel.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/original',
                                            trust_remote_code=True,
                                            torch_dtype=torch.bfloat16).cuda()
    return llm


def run_inference_test(llm):
    llm = FastLanguageModel.for_inference(llm)

    start_at = time.time()

    for i in range(TEST_PROMPT_COUNT):
        print(f'testing for prompt {i} ...')

        test_prompt = dataset_df['input_data'][START_PROMPT_IDX + i]
        inputs = tokenizer(test_prompt, return_tensors='pt').to(llm.device)

        outputs = llm.generate(**inputs, max_length=80, do_sample=True)
        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        _ = llm_answer[len(test_prompt):]

    elapsed_time = time.time() - start_at
    print(f'[INFERENCE] elapsed time with Unsloth (sec.) : {elapsed_time}')


if __name__ == '__main__':

    # 1. read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/OhLoRA_fine_tuning.csv')

    # 2. get LLM
    llm = get_llm()
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    used_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f'used memory (inference) : {used_memory:.2f} MB')

    # 3. inference test
    run_inference_test(llm)

    # 4. Fine-Tuning test prepare
    lora_llm = get_lora_llm(llm=llm, lora_rank=64)
    dataset_df['text'] = dataset_df.apply(lambda x: f"{x['input_data']} ### Answer: {x['output_data']}", axis=1)
    dataset = generate_llm_trainable_dataset(dataset_df)

    response_template = [43774, 10358, 235292]  # '### Answer :'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = get_training_args(with_unsloth=True)
    trainer = get_sft_trainer(lora_llm, dataset, tokenizer, collator, training_args)

    llm = FastLanguageModel.for_training(llm, use_gradient_checkpointing='unsloth')

    # 5. Fine-Tuning test
    fine_tuning_start_at = time.time()
    trainer.train()

    elapsed_time = time.time() - fine_tuning_start_at
    used_memory = torch.cuda.memory_allocated() / (1024 * 1024)

    print(f'used memory (Fine-Tuning, LoRA rank = 64) : {used_memory:.2f} MB')
    print(f'[FINE-TUNING] elapsed time with Unsloth (sec.) : {elapsed_time}')

    trainer.save_model(output_dir_path_with_unsloth)
