# for LLM answer generation config :
# - https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig
from fine_tuning.utils import get_instruction, koreanlm_tokenize

import pandas as pd
import torch

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# stop when "LAST N TOKENS MATCHES stop_token_ids" - class code by ChatGPT-4o
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = list(stop_token_ids.detach().cpu().numpy())
        self.current_ids = []

    def __call__(self, input_ids, scores, **kwargs):
        self.current_ids = input_ids[0].tolist()

        if len(self.current_ids) >= len(self.stop_token_ids):
            if self.current_ids[-len(self.stop_token_ids):] == self.stop_token_ids:
                return True  # stop generation

        return False


# Fine Tuning 된 LLM 을 이용한 inference 실시
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - LLM 4개 개별적 학습 로직에 맞게, user_prompt -> final_input_prompt 로 변수명 수정
# - output column name 에 따라 empty answer 필터링 여부를 다르게 결정

# Arguments:
# - fine_tuned_llm        (LLM)           : Fine-Tuning 된 LLM
# - final_input_prompt    (str)           : LLM 에 최종 입력되는 프롬프트 (경우에 따라 사용자 프롬프트 + alpha)
# - tokenizer             (AutoTokenizer) : LLM 의 Tokenizer
# - output_col            (str)           : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name
# - answer_start_mark     (str)           : 질문의 맨 마지막에 오는 '(답변 시작)' 과 같은 문구 (LLM이 답변을 하도록 유도 목적)
# - stop_token_list       (list)          : stopping token ('(답변 종료)') 에 해당하는 token 의 list
# - max_trials            (int)           : LLM 이 empty answer 가 아닌 답변을 출력하도록 하는 최대 시도 횟수
# - remove_token_type_ids (bool)          : tokenizer 로 Encoding 된 input 의 dict 에서 'token_type_ids' 제거 여부

# Returns:
# - llm_answer       (str) : LLM 답변 중 user prompt 를 제외한 부분
# - trial_count      (int) : LLM 이 empty answer 가 아닌 답변을 출력하기까지의 시도 횟수
# - output_token_cnt (int) : LLM output 의 token 개수

def run_inference(fine_tuned_llm, final_input_prompt, tokenizer, output_col, answer_start_mark, stop_token_list,
                  max_trials=30, remove_token_type_ids=False):

    final_input_prompt_ = final_input_prompt + answer_start_mark

    if remove_token_type_ids:
        inputs = tokenizer(final_input_prompt_, return_tensors='pt')
        inputs = {'input_ids': inputs['input_ids'].to(fine_tuned_llm.device),
                  'attention_mask': inputs['attention_mask'].to(fine_tuned_llm.device)}
    else:
        inputs = tokenizer(final_input_prompt_, return_tensors='pt').to(fine_tuned_llm.device)

    llm_answer = ''
    trial_count = 0
    output_token_cnt = None

    # for stopping criteria
    stop_token_ids = torch.tensor(stop_token_list).to(fine_tuned_llm.device)  # '(답변 종료)'
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    while trial_count < max_trials:
        outputs = fine_tuned_llm.generate(**inputs,
                                          max_length=80,
                                          do_sample=True,
                                          temperature=0.6,
                                          stopping_criteria=stopping_criteria)
        output_token_cnt = len(outputs[0])

        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_input_prompt_):]
        trial_count += 1

        # check LLM answer and return or retry
        is_bracketed = llm_answer.startswith('[') and llm_answer.endswith(']')
        is_non_empty = (not is_bracketed) and llm_answer.replace('\n', '').replace('(답변 종료)', '').replace(' ', '') != ''

        if (is_non_empty or output_col == 'memory') and 'http' not in llm_answer:
            break

    # remove new-lines
    llm_answer = llm_answer.replace('\n', '')

    return llm_answer, trial_count, output_token_cnt


# Fine Tuning 된 LLM 을 이용한 inference 실시 (with KoreanLM-1.5B)
# Create Date : 2025.05.12
# Last Update Date : 2025.05.13
# - LLM 4개 개별적 학습 로직에 맞게, user_prompt -> final_input_prompt 로 변수명 수정

# Arguments:
# - fine_tuned_llm     (LLM)           : Fine-Tuning 된 LLM (KoreanLM-1.5B)
# - final_input_prompt (str)           : LLM 에 입력할 사용자 프롬프트
# - tokenizer          (AutoTokenizer) : LLM 의 Tokenizer
# - prompter           (Prompter)      : LLM 의 사용자 prompt & 답변의 입출력 형식을 나타내는 객체
# - max_trials         (int)           : LLM 이 empty answer 가 아닌 답변을 출력하도록 하는 최대 시도 횟수

# Returns:
# - llm_answer       (str) : LLM 답변 중 user prompt 를 제외한 부분
# - trial_count      (int) : LLM 이 empty answer 가 아닌 답변을 출력하기까지의 시도 횟수
# - output_token_cnt (int) : LLM output 의 token 개수

def run_inference_koreanlm(fine_tuned_llm, final_input_prompt, tokenizer, prompter, max_trials=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    instruction = get_instruction()

    input_ = prompter.generate_prompt(instruction, final_input_prompt)
    tokenized_input = koreanlm_tokenize(input_, tokenizer, return_tensors='pt')
    input_ids = tokenized_input['input_ids'].to(device)

    generation_config = GenerationConfig(temperature=0.6,
                                         top_p=0.9,
                                         top_k=50,
                                         num_beams=1)

    llm_answer = ''
    trial_count = 0
    output_token_cnt = None

    while trial_count < max_trials:
        outputs = fine_tuned_llm.generate(input_ids=input_ids,
                                          generation_config=generation_config,
                                          return_dict_in_generate=True,
                                          output_scores=True,
                                          max_new_tokens=80)
        output_token_cnt = len(outputs.sequences[0])

        llm_answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(tokenized_input):]
        trial_count += 1

        # check LLM answer and return or retry
        is_bracketed = llm_answer.startswith('[') and llm_answer.endswith(']')
        is_non_empty = (not is_bracketed) and llm_answer.replace('\n', '').replace('(답변 종료)', '').replace(' ', '') != ''

        if is_non_empty and 'http' not in llm_answer:
            break

    # remove new-lines
    llm_answer = llm_answer.replace('\n', '')

    return llm_answer, trial_count, output_token_cnt
