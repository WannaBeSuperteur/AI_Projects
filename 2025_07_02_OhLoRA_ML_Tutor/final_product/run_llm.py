import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


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


# LLM 별 Stop Token List 반환
# Create Date : 2025.09.22
# Last Update Date : -

# Arguments :
# - function_type (str) : 실행할 기능으로, 'qna', 'quiz', 'interview' 중 하나

def get_stop_token_list(function_type):
    if function_type == 'qna':
        return [109659, 104449, 99458, 64356]  # (답변 종료)
    elif function_type == 'quiz':
        return [34983, 102546, 99458, 64356]  # (해설 종료)
    else:  # interview
        return [102133, 57390, 99458, 64356]  # (발화 종료)


# Oh-LoRA (오로라) 의 답변 생성
# Create Date : 2025.09.24
# Last Update Date : -

# Arguments :
# - ohlora_llm           (LLM)       : output_message LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned)
# - ohlora_llm_tokenizer (tokenizer) : output_message LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned) 의 tokenizer
# - final_ohlora_input   (str)       : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 summary, memory text 포함)
# - function_type        (str)       : 실행할 기능으로, 'qna', 'quiz', 'interview' 중 하나

# Returns :
# - ohlora_answer (str) : 오로라👱‍♀️ 가 생성한 답변

def generate_llm_answer(ohlora_llm, ohlora_llm_tokenizer, final_ohlora_input, function_type):
    trial_count = 0
    max_trials = 5

    # tokenize final Oh-LoRA input
    if function_type == 'qna':
        answer_start_mark = '(답변 시작)'
        answer_end_mark = '(답변 종료)'
    elif function_type == 'quiz':
        answer_start_mark = '(해설 시작)'
        answer_end_mark = '(해설 종료)'
    else:  # interview
        answer_start_mark = '(발화 시작)'
        answer_end_mark = '(발화 종료)'

    final_ohlora_input_ = final_ohlora_input + ' ' + answer_start_mark

    inputs = ohlora_llm_tokenizer(final_ohlora_input_, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(ohlora_llm.device),
              'attention_mask': inputs['attention_mask'].to(ohlora_llm.device)}

    # for stopping criteria
    stop_token_ids = torch.tensor(get_stop_token_list(function_type=function_type)).to(ohlora_llm.device)
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    while trial_count < max_trials:
        outputs = ohlora_llm.generate(**inputs,
                                      max_length=512,
                                      do_sample=True,
                                      temperature=0.6,
                                      stopping_criteria=stopping_criteria)

        llm_answer = ohlora_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_ohlora_input_):]
        llm_answer = llm_answer.replace('\u200b', '').replace('\xa0', '')  # zwsp, nbsp (폭 없는 공백, 줄 바꿈 없는 공백) 제거

        trial_count += 1

        # check LLM answer and return or retry
        is_empty = llm_answer.replace('\n', '').replace('(발화 종료)', '').replace(' ', '') == ''
        is_answer_end_mark = ('발화 종료' in llm_answer.replace('(발화 종료', '') or
                              '답변종료' in llm_answer.replace('(답변 종료', '') or
                              '해설종료' in llm_answer.replace('(해설 종료', ''))

        is_other_mark = '(사용자' in llm_answer.replace(' ', '') or '요약)' in llm_answer.replace(' ', '')
        is_low_quality = is_empty or is_answer_end_mark or is_other_mark

        if not is_low_quality and ('http' not in llm_answer):
            return llm_answer.replace(answer_end_mark, '')

    return '(읽씹)'


# Oh-LoRA (오로라) 의 답변을 clean 처리
# Create Date : 2025.09.22
# Last Update Date : -

# Arguments :
# - ohlora_answer (str) : 오로라👱‍♀️ 가 생성한 답변

# Returns :
# - llm_answer_cleaned (str) : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

def clean_llm_answer(ohlora_answer):
    return ohlora_answer
