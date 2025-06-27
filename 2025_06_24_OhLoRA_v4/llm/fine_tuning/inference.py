
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from fine_tuning.utils import get_temperature, get_answer_end_mark
except:
    from llm.fine_tuning.utils import get_temperature, get_answer_end_mark

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


# Fine Tuning 된 LLM 을 이용한 inference 실시 (Polyglot-Ko 1.3B)
# Create Date : 2025.05.31
# Last Update Date : -

# Arguments:
# - fine_tuned_llm     (LLM)           : Fine-Tuning 된 LLM
# - final_input_prompt (str)           : LLM 에 최종 입력되는 프롬프트 (경우에 따라 사용자 프롬프트 + alpha)
# - tokenizer          (AutoTokenizer) : LLM 의 Tokenizer
# - output_col         (str)           : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name
# - answer_start_mark  (str)           : 질문의 맨 마지막에 오는 '(답변 시작)' 과 같은 문구 (LLM이 답변을 하도록 유도 목적)
# - stop_token_list    (list)          : stopping token ('(답변 종료)', '(요약 종료)' 등) 에 해당하는 token 의 list
# - max_trials         (int)           : LLM 이 empty answer 가 아닌 답변을 출력하도록 하는 최대 시도 횟수

# Returns:
# - llm_answer       (str) : LLM 답변 중 user prompt 를 제외한 부분
# - trial_count      (int) : LLM 이 empty answer 가 아닌 답변을 출력하기까지의 시도 횟수
# - output_token_cnt (int) : LLM output 의 token 개수

def run_inference_polyglot(fine_tuned_llm, final_input_prompt, tokenizer, output_col, answer_start_mark,
                           stop_token_list, max_trials=30):

    final_input_prompt_ = final_input_prompt + answer_start_mark

    inputs = tokenizer(final_input_prompt_, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(fine_tuned_llm.device),
              'attention_mask': inputs['attention_mask'].to(fine_tuned_llm.device)}

    llm_answer = ''
    trial_count = 0
    output_token_cnt = None

    # for stopping criteria
    stop_token_ids = torch.tensor(stop_token_list).to(fine_tuned_llm.device)  # '(답변 종료)', '(요약 종료)' 등
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    answer_end_mark = get_answer_end_mark(output_col)

    if output_col == 'summary':
        max_length = 192
    else:
        max_length = 128

    while trial_count < max_trials:
        outputs = fine_tuned_llm.generate(**inputs,
                                          max_length=max_length,
                                          do_sample=True,
                                          temperature=get_temperature(output_col, llm_name='polyglot'),
                                          stopping_criteria=stopping_criteria)
        output_token_cnt = len(outputs[0])

        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_input_prompt_):]
        trial_count += 1

        # check LLM answer and return or retry
        is_bracketed = llm_answer.startswith('[') and llm_answer.endswith(']')
        is_non_empty = (not is_bracketed) and llm_answer.replace('\n', '').replace(answer_end_mark, '').replace(' ', '') != ''

        if (is_non_empty or output_col == 'memory') and 'http' not in llm_answer:
            break

    # remove new-lines
    llm_answer = llm_answer.replace('\n', '')

    return llm_answer, trial_count, output_token_cnt


# Fine Tuning 된 LLM 을 이용한 inference 실시 (Kanana-1.5 2.1B)
# Create Date : 2025.05.31
# Last Update Date : 2025.06.04
# - Kanana-1.5-2.1B instruct LLM (kananai) 옵션 추가에 따른 instruct_version 변수 추가

# Arguments:
# - fine_tuned_llm     (LLM)           : Fine-Tuning 된 LLM
# - final_input_prompt (str)           : LLM 에 최종 입력되는 프롬프트 (경우에 따라 사용자 프롬프트 + alpha)
# - tokenizer          (AutoTokenizer) : LLM 의 Tokenizer
# - output_col         (str)           : 학습 데이터 csv 파일의 LLM output 에 해당하는 column name
# - answer_start_mark  (str)           : 질문의 맨 마지막에 오는 '(답변 시작)' 과 같은 문구 (LLM이 답변을 하도록 유도 목적)
# - stop_token_list    (list)          : stopping token ('(답변 종료)', '(요약 종료)' 등) 에 해당하는 token 의 list
# - max_trials         (int)           : LLM 이 empty answer 가 아닌 답변을 출력하도록 하는 최대 시도 횟수
# - instruct_version   (bool)          : True for Kanana-1.5-2.1B instruct, False for Kanana-1.5-2.1B base

# Returns:
# - llm_answer       (str) : LLM 답변 중 user prompt 를 제외한 부분
# - trial_count      (int) : LLM 이 empty answer 가 아닌 답변을 출력하기까지의 시도 횟수
# - output_token_cnt (int) : LLM output 의 token 개수

def run_inference_kanana(fine_tuned_llm, final_input_prompt, tokenizer, output_col, answer_start_mark,
                         stop_token_list, instruct_version, max_trials=30):

    kanana_llm_name = 'kananai' if instruct_version else 'kanana'

    tokenizer.pad_token = tokenizer.eos_token
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    final_input_prompt_ = final_input_prompt + answer_start_mark
    inputs = tokenizer(final_input_prompt_, return_tensors='pt').to(fine_tuned_llm.device)

    llm_answer = ''
    trial_count = 0
    output_token_cnt = None

    # for stopping criteria
    stop_token_ids = torch.tensor(stop_token_list).to(fine_tuned_llm.device)  # '(답변 종료)', '(요약 종료)' 등
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    answer_end_mark = get_answer_end_mark(output_col)

    if output_col == 'summary':
        max_length = 192
    else:
        max_length = 128

    while trial_count < max_trials:
        outputs = fine_tuned_llm.generate(**inputs,
                                          max_length=max_length,
                                          do_sample=True,
                                          temperature=get_temperature(output_col, llm_name=kanana_llm_name),
                                          stopping_criteria=stopping_criteria)
        output_token_cnt = len(outputs[0])

        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_input_prompt_):]
        trial_count += 1

        # check LLM answer and return or retry
        is_bracketed = llm_answer.startswith('[') and llm_answer.endswith(']')
        is_non_empty = (not is_bracketed) and llm_answer.replace('\n', '').replace(answer_end_mark, '').replace(' ', '') != ''

        if (is_non_empty or output_col == 'memory') and 'http' not in llm_answer:
            break

    # remove new-lines
    llm_answer = llm_answer.replace('\n', '')

    return llm_answer, trial_count, output_token_cnt
