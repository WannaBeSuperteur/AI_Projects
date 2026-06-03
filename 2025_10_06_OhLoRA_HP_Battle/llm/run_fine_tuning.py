import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # remove for LLM inference only

import pandas as pd
import torch
import time

import peft
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainingArguments, TrainerState, \
                         TrainerControl

try:
    from utils import load_valid_final_prompts, get_answer_start_mark, get_answer_end_mark, get_stop_token_list, \
                      get_temperature, preview_dataset, add_train_log, add_inference_log
    from inference import run_inference_kanana
except:
    from llm.utils import load_valid_final_prompts, get_answer_start_mark, get_answer_end_mark, get_stop_token_list, \
                          get_temperature, preview_dataset, add_train_log, add_inference_log
    from llm.inference import run_inference_kanana


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ORIGINAL_LLM_PATH = f'{PROJECT_DIR_PATH}/llm/original_models/kananai_original'
FINE_TUNED_LLM_PATH = f'{PROJECT_DIR_PATH}/llm/models/kananai_sft_final_fine_tuned'
ANSWER_CNT = 4
ANSWER_END_MARK = get_answer_end_mark()


lora_llm = None
tokenizer = None
valid_final_prompts = None

train_log_dict = {'epoch': [], 'time': [], 'loss': [], 'grad_norm': [], 'learning_rate': [], 'mean_token_accuracy': []}
inference_log_dict = {'epoch': [], 'elapsed_time (s)': [], 'prompt': [], 'llm_answer': [],
                      'trial_cnt': [], 'output_tkn_cnt': []}

log_dir_path = f'{PROJECT_DIR_PATH}/llm/fine_tuning_logs'
os.makedirs(log_dir_path, exist_ok=True)


class OhLoRACustomCallback(TrainerCallback):

    def __init__(self):
        super(OhLoRACustomCallback, self).__init__()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer, valid_final_prompts

        train_log_df = pd.DataFrame(train_log_dict)
        train_log_df.to_csv(f'{log_dir_path}/kananai_sft_final_train_log.csv')

        print('=== INFERENCE TEST ===')

        for final_input_prompt in valid_final_prompts:
            start_at = time.time()
            stop_token_list = get_stop_token_list()
            answer_start_mark = get_answer_start_mark()

            llm_answer, trial_count, output_token_cnt = run_inference_kanana(lora_llm,
                                                                             final_input_prompt,
                                                                             tokenizer,
                                                                             stop_token_list=stop_token_list,
                                                                             answer_start_mark=answer_start_mark)
            llm_answer = llm_answer[:-len(ANSWER_END_MARK) + 1]
            elapsed_time = time.time() - start_at

            print(f'final input prompt : {final_input_prompt}')
            print(f'llm answer (trials: {trial_count}, output tkns: {output_token_cnt}) : {llm_answer}')

            inference_result = {'epoch': state.epoch, 'elapsed_time': elapsed_time, 'prompt': final_input_prompt,
                                'llm_answer': llm_answer, 'trial_cnt': trial_count, 'output_tkn_cnt': output_token_cnt}
            add_inference_log(inference_result, inference_log_dict)

        inference_log_df = pd.DataFrame(inference_log_dict)
        inference_log_df.to_csv(f'{log_dir_path}/kananai_sft_final_inference_log_dict.csv')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            add_train_log(state, train_log_dict)
        except Exception as e:
            print(f'logging failed : {e}')


# Original LLM (Kanana-1.5 2.1B Instruct) 에 대한 LoRA (Low-Rank Adaption) 적용된 LLM 가져오기
# Create Date : 2026.06.03
# Last Update Date : -

# Arguments:
# - llm       (LLM) : Fine-Tuning 실시할 LLM (Kanana-1.5 2.1B)
# - lora_rank (int) : LoRA 적용 시의 Rank

# Returns:
# - lora_llm (LLM) : LoRA 가 적용된 LLM

def get_lora_llm(llm, lora_rank):
    global lora_llm

    # Kanana-1.5 is based on LlamaForCausalLM architecture
    # target modules of LlamaForCausalLM : ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_rank,                          # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,                    # Dropout for LoRA
        init_lora_weights="gaussian",         # LoRA weight initialization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    lora_llm = peft.get_peft_model(llm, lora_config)
    lora_llm.print_trainable_parameters()


# Original LLM (Kanana-1.5 2.1B Instruct) 에 대한 Fine-Tuning 을 위한 Training Arguments 가져오기
# Create Date : 2026.06.03
# Last Update Date : -

# Returns:
# - training_args (SFTConfig) : Training Arguments

def get_training_args(num_train_epochs):
    training_args = SFTConfig(
        learning_rate=0.0003,               # lower learning rate is recommended for Fine-Tuning
        num_train_epochs=num_train_epochs,
        logging_steps=5,                    # logging frequency
        gradient_checkpointing=False,
        output_dir=FINE_TUNED_LLM_PATH,
        save_total_limit=3,                 # max checkpoint count to save
        per_device_train_batch_size=2,      # batch size per device during training
        per_device_eval_batch_size=1,       # batch size per device during validation
        report_to=None                      # to prevent wandb API key request at start of Fine-Tuning
    )

    return training_args


# Original LLM (Kanana-1.5 2.1B Instruct) 가져오기 (Fine-Tuning 실시할)
# Create Date : 2026.06.03
# Last Update Date : -

# Returns:
# - original_llm (LLM) : Original Kanana-1.5 2.1B Instruct LLM

def get_original_llm():
    original_llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=ORIGINAL_LLM_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).cuda()

    print(f'Original LLM load successful : {ORIGINAL_LLM_PATH}')
    return original_llm


# Original LLM (Kanana-1.5 2.1B Instruct) 에 대한 Fine-Tuning 을 위한 SFT (Supervised Fine-Tuning) Trainer 가져오기
# Create Date : 2026.06.03
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
        callbacks=[OhLoRACustomCallback()]
    )

    return trainer


# Original LLM (Kanana-1.5 2.1B Instruct) 에 대한 LLM 이 직접 학습 가능한 데이터셋 가져오기
# Create Date : 2026.06.03
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 데이터가 저장된 DataFrame

# Returns:
# - dataset (Dataset) : LLM 학습 데이터셋

def generate_llm_trainable_dataset(dataset_df):
    global tokenizer

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'].str.startswith('train')][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'].str.startswith('valid')][['text']])
    preview_dataset(dataset, tokenizer)

    return dataset


# LLM (Kanana-1.5 2.1B Instruct) Fine-Tuning 실시
# Create Date : 2026.06.03
# Last Update Date : -

# Returns:
# - llm/models/kananai_sft_final_fine_tuned 에 Fine-Tuning 된 모델 저장

def fine_tune_kanana():
    global lora_llm, tokenizer, valid_final_prompts
    valid_final_prompts = load_valid_final_prompts()

    print('Oh-LoRA LLM Fine Tuning start.')

    # get original LLM and tokenizer
    # Kanana-1.5 2.1B original model is from https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505
    original_llm = get_original_llm()
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_LLM_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    original_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    # read dataset
    dataset_df = pd.read_csv(f'{PROJECT_DIR_PATH}/llm/train_data.csv')
    dataset_df = dataset_df.sample(frac=1)  # shuffle

    # prepare Fine-Tuning
    get_lora_llm(llm=original_llm, lora_rank=64)

    dataset_df['text'] = dataset_df.apply(
        lambda x: f"{x['input_data']} (답변 시작) ### 답변: {x['output_message']} (답변 종료) <|end_of_text|>",
        axis=1)
    dataset = generate_llm_trainable_dataset(dataset_df)
    preview_dataset(dataset, tokenizer)

    response_template = [8, 17010, 111964, 25]  # '### 답변 :'

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    training_args = get_training_args(num_train_epochs=5)
    trainer = get_sft_trainer(dataset, collator, training_args)

    # run Fine-Tuning
    trainer.train()

    # save Fine-Tuned model
    trainer.save_model(FINE_TUNED_LLM_PATH)


# Fine-Tuning 된 LLM 로딩
# Create Date : 2026.06.03
# Last Update Date : -

# Returns:
# - fine_tuned_llm (LLM) : Fine-Tuning 된 LLM

def load_fine_tuned_llm():
    fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
        FINE_TUNED_LLM_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).cuda()

    return fine_tuned_llm


# LLM inference (해당 LLM 이 없거나 로딩 실패 시 Fine-Tuning 학습) 실시
# Create Date : 2026.06.03
# Last Update Date : -

def inference_or_fine_tune_llm():

    # load valid dataset
    valid_final_input_prompts = load_valid_final_prompts()

    for final_input_prompt in valid_final_input_prompts:
        print(f'final input prompt for validation : {final_input_prompt}')

    # try load LLM -> when failed, run Fine-Tuning and save LLM
    try:
        fine_tuned_llm = load_fine_tuned_llm()
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_LLM_PATH)
        print(f'Fine-Tuned LLM - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'Fine-Tuned LLM load failed : {e}')
        fine_tune_kanana()

        fine_tuned_llm = load_fine_tuned_llm()
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_LLM_PATH)

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    inference_temperature = get_temperature()
    inference_log_path = f'{log_dir_path}/kananai_sft_final_inference_log_final_{inference_temperature}.txt'
    inference_log = ''

    # run inference using Fine-Tuned LLM
    for final_input_prompt in valid_final_input_prompts:
        llm_input_print = f'\nLLM input :\n{final_input_prompt}'
        print(llm_input_print)

        # generate 4 answers for comparison
        llm_answers = []
        trial_counts = []
        output_token_cnts = []
        elapsed_times = []

        for _ in range(ANSWER_CNT):
            answer_start_mark = get_answer_start_mark()

            inference_start_at = time.time()
            stop_token_list = get_stop_token_list()
            llm_answer, trial_count, output_token_cnt = run_inference_kanana(fine_tuned_llm,
                                                                             final_input_prompt,
                                                                             tokenizer,
                                                                             stop_token_list=stop_token_list,
                                                                             answer_start_mark=answer_start_mark)

            elapsed_time = time.time() - inference_start_at

            llm_answer = llm_answer[:-len(ANSWER_END_MARK) + 1]
            llm_answers.append(llm_answer)
            trial_counts.append(str(trial_count))
            output_token_cnts.append(str(output_token_cnt))
            elapsed_times.append(elapsed_time)

        trial_counts_str = ','.join(trial_counts)
        output_token_cnts_str = ','.join(output_token_cnts)
        llm_answers_and_times = [f'{llm_answers[i]} (🕚 {round(elapsed_times[i], 2)} s)' for i in range(ANSWER_CNT)]
        llm_answers_and_times_str = '\n- '.join(llm_answers_and_times)

        llm_output_print = (f'Oh-LoRA answer (trials: {trial_counts_str} | output_tkn_cnt : {output_token_cnts_str}) '
                            f':\n- {llm_answers_and_times_str}')
        print(llm_output_print)

        # write inference log
        inference_log += '\n' + llm_input_print + '\n' + llm_output_print

        f = open(inference_log_path, 'w', encoding='UTF8')
        f.write(inference_log)
        f.close()


if __name__ == '__main__':
    print(f'\n=== 🚀 Fine-Tune LLM START 🚀 ===')
    inference_or_fine_tune_llm()
