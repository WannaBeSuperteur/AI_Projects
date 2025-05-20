import torch
from transformers import StoppingCriteriaList

from llm.fine_tuning.inference import StopOnTokens


# Oh-LoRA (오로라) 의 답변 생성
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments :
# - ohlora_llm           (LLM)       : output_message LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - ohlora_llm_tokenizer (tokenizer) : output_message LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
# - final_ohlora_input   (str)       : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 summary, memory text 포함)

# Returns :
# - ohlora_answer (str) : 오로라👱‍♀️ 가 생성한 답변

def generate_llm_answer(ohlora_llm, ohlora_llm_tokenizer, final_ohlora_input):
    trial_count = 0
    max_trials = 30

    # tokenize final Oh-LoRA input
    final_ohlora_input_ = final_ohlora_input + ' (답변 시작)'

    inputs = ohlora_llm_tokenizer(final_ohlora_input_, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(ohlora_llm.device),
              'attention_mask': inputs['attention_mask'].to(ohlora_llm.device)}

    # for stopping criteria
    stop_token_ids = torch.tensor([1477, 1078, 4833, 12]).to(ohlora_llm.device)  # '(답변 종료)'
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    while trial_count < max_trials:
        outputs = ohlora_llm.generate(**inputs,
                                      max_length=96,
                                      do_sample=True,
                                      temperature=0.6,
                                      stopping_criteria=stopping_criteria)

        llm_answer = ohlora_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_ohlora_input_):]
        llm_answer = llm_answer.replace('\u200b', '').replace('\xa0', '')  # zwsp, nbsp (폭 없는 공백, 줄 바꿈 없는 공백) 제거
        trial_count += 1

        # check LLM answer and return or retry
        is_empty = llm_answer.replace('\n', '').replace('(답변 종료)', '').replace(' ', '') == ''
        is_answer_end_mark = '답변 종료' in llm_answer.replace('(답변 종료)', '') or '답변종료' in llm_answer.replace('(답변 종료)', '')
        starts_with_nonblank_in_early_try = trial_count < 3 and not llm_answer.startswith(' ')
        too_many_tokens = len(outputs[0]) >= 91
        is_uncleaned = is_empty or is_answer_end_mark or starts_with_nonblank_in_early_try or too_many_tokens

        is_unnecessary_quote = '"' in llm_answer or '”' in llm_answer or '“' in llm_answer or '’' in llm_answer
        is_unnecessary_mark = '�' in llm_answer
        is_too_many_blanks = '     ' in llm_answer
        is_low_quality = is_unnecessary_quote or is_unnecessary_mark or is_too_many_blanks

        print(trial_count, llm_answer)

        if (not is_uncleaned) and (not is_low_quality) and ('http' not in llm_answer):
            return llm_answer.replace('(답변 종료)', '')

    return '(읽씹)'


# Oh-LoRA (오로라) 의 답변을 clean 처리
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments :
# - ohlora_answer (str) : 오로라👱‍♀️ 가 생성한 답변

# Returns :
# - llm_answer_cleaned (str) : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

def clean_llm_answer(ohlora_answer):
    return ohlora_answer  # cleaning not needed


# Oh-LoRA (오로라) 의 생성된 답변으로부터 memory 정보를 parsing
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments :
# - memory_llm           (LLM)       : memory LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - memory_llm_tokenizer (tokenizer) : memory LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
# - final_ohlora_input   (str) : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 summary, memory text 포함)

# Returns :
# - memory_list (list(str)) : 오로라👱‍♀️ 가 저장해야 할 메모리 목록

def parse_memory(memory_llm, memory_llm_tokenizer, final_ohlora_input):
    raise NotImplementedError


# Oh-LoRA (오로라) 의 메모리 정보를 llm/memory_mechanism/saved_memory/ohlora_memory.txt 에 저장
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments :
# - memory_list (list(str)) : 오로라👱‍♀️ 가 저장해야 할 메모리 목록

def save_memory_list(memory_list):
    raise NotImplementedError


# Oh-LoRA (오로라) 의 답변 요약
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments :
# - summary_llm           (LLM)       : summary LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - summary_llm_tokenizer (tokenizer) : summary LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
# - final_ohlora_input    (str)       : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 summary, memory text 포함)
# - llm_answer_cleaned    (str)       : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

# Returns :
# - llm_summary (str) : 직전 대화 내용에 대한 요약

def summarize_llm_answer(summary_llm, summary_llm_tokenizer, final_ohlora_input, llm_answer_cleaned):
    raise NotImplementedError


# Oh-LoRA (오로라) 의 답변에 따라 눈을 뜬 정도 (eyes), 입을 벌린 정도 (mouth), 고개 돌림 (pose) 점수 산출
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments :
# - eyes_mouth_pose_llm           (LLM)       : eyes_mouth_pose LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - eyes_mouth_pose_llm_tokenizer (tokenizer) : eyes_mouth_pose LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
# - llm_answer_cleaned            (str)       : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

# Returns :
# - eyes_score  (float) : 눈을 뜬 정도 (eyes) 의 속성 값 점수
# - mouth_score (float) : 입을 벌린 정도 (mouth) 의 속성 값 점수
# - pose_score  (float) : 고개 돌림 (pose) 의 속성 값 점수

def decide_property_scores(eyes_mouth_pose_llm, eyes_mouth_pose_llm_tokenizer, llm_answer_cleaned):
    raise NotImplementedError

