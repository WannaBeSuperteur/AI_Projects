
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from llm_fine_tuning.utils import get_temperature, get_answer_end_mark
except:
    from ai_interview.llm_fine_tuning.utils import get_temperature, get_answer_end_mark

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


# Fine Tuning 된 LLM 을 이용한 inference 실시 (Kanana-1.5 2.1B)
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - fine_tuned_llm     (LLM)           : Fine-Tuning 된 LLM
# - final_input_prompt (str)           : LLM 에 최종 입력되는 프롬프트 (퀴즈 + 모범 답안 + 사용자 답안)
# - tokenizer          (AutoTokenizer) : LLM 의 Tokenizer
# - answer_start_mark  (str)           : 질문의 맨 마지막에 오는 '(발화 시작)' 과 같은 문구 (LLM이 답변을 하도록 유도 목적)
# - stop_token_list    (list)          : stopping token, 즉 '(발화 종료)' 에 해당하는 token 의 list
# - max_trials         (int)           : LLM 이 empty answer 가 아닌 답변을 출력하도록 하는 최대 시도 횟수

# Returns:
# - llm_answer       (str) : LLM 답변 중 user prompt 를 제외한 부분
# - trial_count      (int) : LLM 이 empty answer 가 아닌 답변을 출력하기까지의 시도 횟수
# - output_token_cnt (int) : LLM output 의 token 개수

def run_inference_kanana(fine_tuned_llm, final_input_prompt, tokenizer, answer_start_mark,
                         stop_token_list, max_trials=30):

    tokenizer.pad_token = tokenizer.eos_token
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    final_input_prompt_ = final_input_prompt + answer_start_mark
    inputs = tokenizer(final_input_prompt_, return_tensors='pt').to(fine_tuned_llm.device)

    llm_answer = ''
    trial_count = 0
    output_token_cnt = None

    # for stopping criteria
    stop_token_ids = torch.tensor(stop_token_list).to(fine_tuned_llm.device)  # (발화 종료)
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    answer_end_mark = get_answer_end_mark()
    max_length = 1024

    while trial_count < max_trials:
        outputs = fine_tuned_llm.generate(**inputs,
                                          max_length=max_length,
                                          do_sample=True,
                                          temperature=get_temperature(),
                                          stopping_criteria=stopping_criteria)
        output_token_cnt = len(outputs[0])

        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_input_prompt_):]
        trial_count += 1

        # check LLM answer and return or retry
        is_bracketed = llm_answer.startswith('[') and llm_answer.endswith(']')
        is_non_empty = (not is_bracketed) and llm_answer.replace('\n', '').replace(answer_end_mark, '').replace(' ', '') != ''

        if is_non_empty and 'http' not in llm_answer:
            break

    # remove new-lines
    llm_answer = llm_answer.replace('\n', '')

    return llm_answer, trial_count, output_token_cnt


# Fine Tuning 된 LLM 을 이용한 inference 실시 (Mi:dm 2.0 Mini, 2.31B)
# Create Date : 2025.07.28
# Last Update Date : -

# Arguments:
# - fine_tuned_llm     (LLM)           : Fine-Tuning 된 LLM
# - final_input_prompt (str)           : LLM 에 최종 입력되는 프롬프트 (퀴즈 + 모범 답안 + 사용자 답안)
# - tokenizer          (AutoTokenizer) : LLM 의 Tokenizer
# - answer_start_mark  (str)           : 질문의 맨 마지막에 오는 '(발화 시작)' 과 같은 문구 (LLM이 답변을 하도록 유도 목적)
# - stop_token_list    (list)          : stopping token, 즉 '(발화 종료)' 에 해당하는 token 의 list
# - max_trials         (int)           : LLM 이 empty answer 가 아닌 답변을 출력하도록 하는 최대 시도 횟수

# Returns:
# - llm_answer       (str) : LLM 답변 중 user prompt 를 제외한 부분
# - trial_count      (int) : LLM 이 empty answer 가 아닌 답변을 출력하기까지의 시도 횟수
# - output_token_cnt (int) : LLM output 의 token 개수

def run_inference_midm(fine_tuned_llm, final_input_prompt, tokenizer, answer_start_mark,
                       stop_token_list, max_trials=30):

    tokenizer.pad_token = tokenizer.eos_token
    fine_tuned_llm.generation_config.pad_token_id = tokenizer.pad_token_id

    final_input_prompt_ = final_input_prompt + answer_start_mark
    inputs = tokenizer(final_input_prompt_, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(fine_tuned_llm.device),
              'attention_mask': inputs['attention_mask'].to(fine_tuned_llm.device)}

    llm_answer = ''
    trial_count = 0
    output_token_cnt = None

    # for stopping criteria
    stop_token_ids = torch.tensor(stop_token_list).to(fine_tuned_llm.device)  # (발화 종료)
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    answer_end_mark = get_answer_end_mark()
    max_length = 1024

    while trial_count < max_trials:
        outputs = fine_tuned_llm.generate(**inputs,
                                          max_length=max_length,
                                          do_sample=True,
                                          temperature=get_temperature(),
                                          stopping_criteria=stopping_criteria)
        output_token_cnt = len(outputs[0])

        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_input_prompt_):]
        trial_count += 1

        # check LLM answer and return or retry
        is_bracketed = llm_answer.startswith('[') and llm_answer.endswith(']')
        is_non_empty = (not is_bracketed) and llm_answer.replace('\n', '').replace(answer_end_mark, '').replace(' ', '') != ''

        if is_non_empty and 'http' not in llm_answer:
            break

    # remove new-lines
    llm_answer = llm_answer.replace('\n', '')

    return llm_answer, trial_count, output_token_cnt
