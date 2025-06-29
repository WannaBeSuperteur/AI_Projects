import torch
from transformers import StoppingCriteriaList

from llm.fine_tuning.inference import StopOnTokens
from llm.fine_tuning.fine_tuning_kanana import get_stop_token_list as get_kanana_stop_token_list
from llm.fine_tuning.fine_tuning_polyglot import get_stop_token_list as get_polyglot_stop_token_list

from datetime import datetime
import os
import random
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Oh-LoRA (오로라) 답변 생성 시 요일 정보 환각 현상 여부 확인
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - ohlora_answer (str) : 오로라👱‍♀️ 가 생성한 답변

# Returns:
# - is_hallucination (bool) : 환각 현상 여부 (환각 현상 발생 시 True)

def is_dow_hallucination(ohlora_answer):

    # get current day-of-week and current month
    dow_mapping = ['월', '화', '수', '목', '금', '토', '일']
    current_dow = datetime.today().weekday()
    current_hour = datetime.now().hour

    if current_hour < 4:
        dow_text = dow_mapping[(current_dow + 6) % 7]
    else:
        dow_text = dow_mapping[current_dow]

    # check time hallucination
    if dow_text in ['일', '월', '화', '수'] and ('내일 주말' in ohlora_answer or '내일은 주말' in ohlora_answer):
        return True

    if dow_text in ['월', '화', '수', '목'] and ('오늘 주말' in ohlora_answer or '오늘은 주말' in ohlora_answer):
        return True

    if dow_text not in ['목', '금'] and '내일부터 주말이' in ohlora_answer:
        return True

    if dow_text not in ['금', '토'] and '오늘부터 주말이' in ohlora_answer:
        return True

    if dow_text != '목' and ('내일 불금' in ohlora_answer or '내일은 불금' in ohlora_answer):
        return True

    if dow_text != '금' and ('오늘 불금' in ohlora_answer or '오늘은 불금' in ohlora_answer):
        return True

    if dow_text not in ['토', '일'] and '내일부터 한 주 시작' in ohlora_answer:
        return True

    if dow_text not in ['일', '월'] and '오늘부터 한 주 시작' in ohlora_answer:
        return True

    for i in range(7):
        if dow_text != dow_mapping[i] and (f'오늘 {dow_mapping[i]}요일' in ohlora_answer or
                                           f'오늘은 {dow_mapping[i]}요일' in ohlora_answer or
                                           f'오늘도 {dow_mapping[i]}요일' in ohlora_answer):
            return True

        if dow_text != dow_mapping[i] and (f'내일 {dow_mapping[(i + 1) % 7]}요일' in ohlora_answer or
                                           f'내일은 {dow_mapping[(i + 1) % 7]}요일' in ohlora_answer or
                                           f'내일도 {dow_mapping[(i + 1) % 7]}요일' in ohlora_answer):
            return True

    # no time hallucination
    return False


# Oh-LoRA (오로라) 의 답변 생성
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - ohlora_llm           (LLM)       : output_message LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned)
# - ohlora_llm_tokenizer (tokenizer) : output_message LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned) 의 tokenizer
# - final_ohlora_input   (str)       : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 summary, memory text 포함)

# Returns :
# - ohlora_answer (str) : 오로라👱‍♀️ 가 생성한 답변

def generate_llm_answer(ohlora_llm, ohlora_llm_tokenizer, final_ohlora_input):
    trial_count = 0
    time_hallucination_count = 0
    max_trials = 5

    # tokenize final Oh-LoRA input
    final_ohlora_input_ = final_ohlora_input + ' (답변 시작)'

    inputs = ohlora_llm_tokenizer(final_ohlora_input_, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(ohlora_llm.device),
              'attention_mask': inputs['attention_mask'].to(ohlora_llm.device)}

    # for stopping criteria
    stop_token_ids = torch.tensor(get_kanana_stop_token_list('output_message')).to(ohlora_llm.device)  # '(답변 종료)'
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    while trial_count < max_trials:
        outputs = ohlora_llm.generate(**inputs,
                                      max_length=128,
                                      do_sample=True,
                                      temperature=0.6 + 0.1 * time_hallucination_count,
                                      stopping_criteria=stopping_criteria)

        llm_answer = ohlora_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_ohlora_input_):]
        llm_answer = llm_answer.replace('\u200b', '').replace('\xa0', '')  # zwsp, nbsp (폭 없는 공백, 줄 바꿈 없는 공백) 제거

        trial_count += 1

        # '오늘는', '내일는' -> '오늘은', '내일은'
        llm_answer = llm_answer.replace('오늘는', '오늘은')
        llm_answer = llm_answer.replace('내일는', '내일은')

        # check LLM answer and return or retry
        is_empty = llm_answer.replace('\n', '').replace('(답변 종료)', '').replace(' ', '') == ''
        is_answer_end_mark = '답변 종료' in llm_answer.replace('(답변 종료)', '') or '답변종료' in llm_answer.replace('(답변 종료)', '')
        is_other_mark = '(사용자' in llm_answer.replace(' ', '') or '요약)' in llm_answer.replace(' ', '')

        is_time_hallucinated = trial_count < max_trials and is_dow_hallucination(llm_answer)
        is_low_quality = (is_empty or is_answer_end_mark or is_other_mark) or is_time_hallucinated

        if is_time_hallucinated:
            time_hallucination_count += 1

        if not is_low_quality and ('http' not in llm_answer):
            return llm_answer.replace('(답변 종료)', '')

    return '(읽씹)'


# Oh-LoRA (오로라) 의 답변을 clean 처리
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - ohlora_answer (str) : 오로라👱‍♀️ 가 생성한 답변

# Returns :
# - llm_answer_cleaned (str) : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

def clean_llm_answer(ohlora_answer):
    return ohlora_answer


# Oh-LoRA (오로라) 의 생성된 답변으로부터 memory 정보를 parsing
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - memory_llm           (LLM)       : memory LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned)
# - memory_llm_tokenizer (tokenizer) : memory LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned) 에 대한 tokenizer
# - final_ohlora_input   (str)       : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 summary, memory text 포함)

# Returns :
# - memory_list (list(str) or None) : 오로라👱‍♀️ 가 저장해야 할 메모리 목록

def parse_memory(memory_llm, memory_llm_tokenizer, final_ohlora_input):
    trial_count = 0
    max_trials = 5

    # tokenize final Oh-LoRA input
    final_ohlora_input_ = final_ohlora_input + ' (요약 시작)'

    inputs = memory_llm_tokenizer(final_ohlora_input_, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(memory_llm.device),
              'attention_mask': inputs['attention_mask'].to(memory_llm.device)}

    # for stopping criteria
    stop_token_ids = torch.tensor(get_polyglot_stop_token_list('memory')).to(memory_llm.device)  # '(요약 종료)'
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    while trial_count < max_trials:
        outputs = memory_llm.generate(**inputs,
                                      max_new_tokens=24,
                                      do_sample=True,
                                      temperature=0.6,
                                      stopping_criteria=stopping_criteria)

        llm_answer = memory_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer.replace('(요약 종료)', '')

        # post-process memory LLM answer
        llm_answer = llm_answer[len(final_ohlora_input_):]
        llm_answer = llm_answer.replace('\u200b', '').replace('\xa0', '')    # zwsp, nbsp (폭 / 줄 바꿈 없는 공백) 제거
        llm_answer = llm_answer.replace('(', '[').replace(')', ']')          # (...) -> [...]
        llm_answer = llm_answer.replace('{', '[').replace('}', ']')          # {...} -> [...]
        llm_answer = llm_answer.replace(' : ', ': ')                         # [key : value] -> [key: value]

        if '[' in llm_answer and ']' in llm_answer:
            llm_answer = '[' + llm_answer.split('[')[1].split(']')[0] + ']'  # blah [key: value] blah -> [key: value]

        if ': ' not in llm_answer:
            llm_answer = llm_answer.replace(':', ': ')           # [key:value] -> [key: value]

        if '[' not in llm_answer and ']' not in llm_answer and ': ' in llm_answer:
            llm_answer = '[' + llm_answer + ']'                              # key: value -> [key: value]

        trial_count += 1

        # check LLM answer and return or retry
        is_answer_end_mark = '요약 종료' in llm_answer or '요약종료' in llm_answer
        too_many_tokens = len(outputs[0]) - len(inputs['input_ids'][0]) == 24

        is_in_format = llm_answer == '' or (llm_answer[0] == '[' and llm_answer[-1] == ']' and ': ' in llm_answer)
        is_in_format_cnts = llm_answer.count('[') == 1 and llm_answer.count(']') == 1 and llm_answer.count(':') == 1

        # empty answer -> format is CORRECT
        if llm_answer.replace(' ', '') == '':
            is_format_correct = True

        # for non-empty answer
        elif is_in_format and is_in_format_cnts:
            is_key_nonempty = len(llm_answer.split('[')[1].split(':')[0].replace(' ', '')) >= 1
            is_value_nonempty = len(llm_answer.split(':')[1].split(']')[0].replace(' ', '')) >= 1
            is_format_correct = is_key_nonempty and is_value_nonempty
        else:
            is_format_correct = False

        is_uncleaned = (is_answer_end_mark or too_many_tokens) or not is_format_correct

        is_unnecessary_mark = '�' in llm_answer
        is_too_many_blanks = '     ' in llm_answer
        is_low_quality = is_unnecessary_mark or is_too_many_blanks

        if (not is_uncleaned) and (not is_low_quality) and ('http' not in llm_answer):
            return [llm_answer]

    return None


# Oh-LoRA (오로라) 의 메모리 정보를 llm/memory_mechanism/saved_memory/ohlora_memory.txt 에 저장
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - memory_list (list(str)) : 오로라👱‍♀️ 가 저장해야 할 메모리 목록

def save_memory_list(memory_list):
    memory_file_path = f'{PROJECT_DIR_PATH}/llm/memory_mechanism/saved_memory/ohlora_memory.txt'

    memory_file = open(memory_file_path, 'r', encoding='UTF8')
    memory_file_lines = memory_file.readlines()
    memory_file.close()

    memory_file_lines = [line.split('\n')[0] for line in memory_file_lines]

    for memory in memory_list:
        is_duplicated = False

        for line in memory_file_lines:
            if memory.replace(' ', '') == line.replace(' ', ''):
                is_duplicated = True
                break

        if not is_duplicated:
            memory_file_lines.append(memory)

    memory_file_lines_to_save = '\n'.join(memory_file_lines)

    memory_file = open(memory_file_path, 'w', encoding='UTF8')
    memory_file.write(memory_file_lines_to_save + '\n')
    memory_file.close()


# Oh-LoRA (오로라) 의 답변 요약
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - summary_llm           (LLM)       : summary LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned)
# - summary_llm_tokenizer (tokenizer) : summary LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned) 의 tokenizer
# - final_ohlora_input    (str)       : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 summary, memory text 포함)
# - llm_answer_cleaned    (str)       : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

# Returns :
# - llm_summary (str) : 직전 대화 내용에 대한 요약

def summarize_llm_answer(summary_llm, summary_llm_tokenizer, final_ohlora_input, llm_answer_cleaned):
    trial_count = 0
    max_trials = 3

    # tokenize final Oh-LoRA input
    final_summary_input = final_ohlora_input + ' (오로라 답변)' + llm_answer_cleaned + ' (요약 시작)'

    inputs = summary_llm_tokenizer(final_summary_input, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(summary_llm.device),
              'attention_mask': inputs['attention_mask'].to(summary_llm.device)}

    # for stopping criteria
    stop_token_ids = torch.tensor(get_kanana_stop_token_list('summary')).to(summary_llm.device)  # '(요약 종료)'
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    while trial_count < max_trials:
        outputs = summary_llm.generate(**inputs,
                                       max_new_tokens=64,
                                       do_sample=True,
                                       temperature=0.6,
                                       stopping_criteria=stopping_criteria)

        llm_answer = summary_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        llm_answer = llm_answer[len(final_summary_input):]
        llm_answer = llm_answer.replace('\u200b', '').replace('\xa0', '')  # zwsp, nbsp (폭 없는 공백, 줄 바꿈 없는 공백) 제거
        llm_answer = llm_answer.strip()

        trial_count += 1

        # check LLM answer and return or retry
        is_answer_end_mark = '요약 종료' in llm_answer.replace('(요약 종료)', '') or '요약종료' in llm_answer.replace('(요약 종료)', '')
        too_many_tokens = len(outputs[0]) - len(inputs['input_ids'][0]) >= 60
        is_almost_empty = len(llm_answer.replace('(요약 종료)', '')) < 2

        if not (is_answer_end_mark or too_many_tokens or is_almost_empty) and ('http' not in llm_answer):
            return llm_answer.replace('(요약 종료)', '')

    return None


# Oh-LoRA (오로라) 의 답변에 따라 눈을 뜬 정도 (eyes), 입을 벌린 정도 (mouth), 고개 돌림 (pose) 텍스트 산출
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - eyes_mouth_pose_llm           (LLM)       : eyes_mouth_pose LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned)
# - eyes_mouth_pose_llm_tokenizer (tokenizer) : eyes_mouth_pose LLM 에 대한 tokenizer
# - llm_answer_cleaned            (str)       : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

# Returns :
# - eyes_score_text  (str) : 눈을 뜬 정도 (eyes) 를 나타내는 텍스트
# - mouth_score_text (str) : 입을 벌린 정도 (mouth) 를 나타내는 텍스트
# - pose_score_text  (str) : 고개 돌림 (pose) 을 나타내는 텍스트

def decide_property_score_texts(eyes_mouth_pose_llm, eyes_mouth_pose_llm_tokenizer, llm_answer_cleaned):
    trial_count = 0
    max_trials = 5

    eyes_score_texts = ['작게', '보통', '크게', '아주 크게']
    mouth_score_texts = ['보통', '크게', '아주 크게']
    pose_score_texts = ['없음', '보통', '많이']

    # tokenize final Oh-LoRA input
    llm_answer_cleaned_ = llm_answer_cleaned + ' (표정 출력 시작)'

    inputs = eyes_mouth_pose_llm_tokenizer(llm_answer_cleaned_, return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids'].to(eyes_mouth_pose_llm.device),
              'attention_mask': inputs['attention_mask'].to(eyes_mouth_pose_llm.device)}

    # for stopping criteria
    eyes_mouth_pose_stop_tokens = get_polyglot_stop_token_list('eyes_mouth_pose')  # '(표정 출력 종료)'
    stop_token_ids = torch.tensor(eyes_mouth_pose_stop_tokens).to(eyes_mouth_pose_llm.device)
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    while trial_count < max_trials:
        outputs = eyes_mouth_pose_llm.generate(**inputs,
                                               max_new_tokens=36,
                                               do_sample=True,
                                               temperature=1.0,
                                               stopping_criteria=stopping_criteria)

        llm_answer = eyes_mouth_pose_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        llm_answer = llm_answer[len(llm_answer_cleaned_):]
        llm_answer = llm_answer.replace('\u200b', '').replace('\xa0', '')  # zwsp, nbsp (폭 없는 공백, 줄 바꿈 없는 공백) 제거
        trial_count += 1

        # check LLM answer and return or retry
        try:
            eyes_score_text = llm_answer.split('눈 ')[1].split(',')[0].split(')')[0]
            mouth_score_text = llm_answer.split('입 ')[1].split(',')[0].split(')')[0]
            pose_score_text = llm_answer.split('고개 돌림 ')[1].split(',')[0].split(')')[0]

            final_eyes_score_text, final_mouth_score_text, final_pose_score_text = None, None, None

            for txt in eyes_score_texts:
                if eyes_score_text.startswith(txt):
                    final_eyes_score_text = txt

            for txt in mouth_score_texts:
                if mouth_score_text.startswith(txt):
                    final_mouth_score_text = txt

            for txt in pose_score_texts:
                if pose_score_text.startswith(txt):
                    final_pose_score_text = txt

            assert final_eyes_score_text is not None
            assert final_mouth_score_text is not None
            assert final_pose_score_text is not None

            return final_eyes_score_text, final_mouth_score_text, final_pose_score_text

        except:
            pass

    eyes_score_text, mouth_score_text, pose_score_text = '보통', '보통', '없음'
    return eyes_score_text, mouth_score_text, pose_score_text


# Oh-LoRA (오로라) 의 답변에 따라 눈을 뜬 정도 (eyes), 입을 벌린 정도 (mouth), 고개 돌림 (pose) 점수 산출
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments :
# - eyes_score_text  (str) : 눈을 뜬 정도 (eyes) 를 나타내는 텍스트
# - mouth_score_text (str) : 입을 벌린 정도 (mouth) 를 나타내는 텍스트
# - pose_score_text  (str) : 고개 돌림 (pose) 을 나타내는 텍스트

# Returns :
# - eyes_score  (float) : 눈을 뜬 정도 (eyes) 를 나타내는 점수 (w vector 에 eyes change vector 를 더할 가중치)
# - mouth_score (float) : 입을 벌린 정도 (mouth) 를 나타내는 점수 (w vector 에 mouth change vector 를 더할 가중치)
# - pose_score  (float) : 고개 돌림 (pose) 을 나타내는 점수 (w vector 에 pose change vector 를 더할 가중치)

def decide_property_scores(eyes_score_text, mouth_score_text, pose_score_text):

    # eyes score mapping
    if eyes_score_text == '작게':
        eyes_score = -0.2 + 0.3 * random.random()

    elif eyes_score_text == '보통':
        eyes_score = 0.3 + 0.3 * random.random()

    elif eyes_score_text == '크게':
        eyes_score = 0.8 + 0.3 * random.random()

    else:  # 아주 크게
        eyes_score = 1.1 + 0.1 * random.random()

    # mouth score mapping
    if mouth_score_text == '보통':
        mouth_score = 0.4 + 0.4 * random.random()

    elif mouth_score_text == '크게':
        mouth_score = 1.0 + 0.4 * random.random()

    else:  # 아주 크게
        mouth_score = 1.4 + 0.2 * random.random()

    # pose score mapping
    if pose_score_text == '없음':
        pose_score = -0.6 + 0.9 * random.random()

    elif mouth_score_text == '보통':
        pose_score = 0.8 + 0.7 * random.random()

    else:  # 많이
        pose_score = 1.5 + 0.3 * random.random()

    # 실제 w vector 에 더해지는 가중치는 "값이 작을수록 (음수 등)" 눈을 크게 뜨고, 입을 크게 벌리고, 고개를 많이 돌림
    eyes_score = (-1.0) * eyes_score
    mouth_score = (-1.0) * mouth_score
    pose_score = (-1.0) * pose_score

    return eyes_score, mouth_score, pose_score
