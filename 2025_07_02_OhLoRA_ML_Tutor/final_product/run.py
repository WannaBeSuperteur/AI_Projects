import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import random
import math
import numpy as np

import argparse
import threading
import os
import sys
import time
from datetime import datetime
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ALL_PROJECTS_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(PROJECT_DIR_PATH)

from run_llm import (generate_llm_answer, clean_llm_answer, parse_memory, save_memory_list, summarize_llm_answer,
                     decide_property_score_texts, decide_property_scores)
from run_display import generate_and_show_ohlora_image

from stylegan.stylegan_common.stylegan_generator import StyleGANGenerator, StyleGANGeneratorForV6
from stylegan.stylegan_vectorfind_v8 import (load_ohlora_z_vectors,
                                             load_ohlora_w_group_names,
                                             get_property_change_vectors)
from ombre.load_seg_modal import load_existing_hair_seg_model

from llm.common import load_pretrained_sbert_model
from llm.memory_mechanism import pick_best_memory_item

import pandas as pd


EYES_BASE_SCORE, MOUTH_BASE_SCORE, POSE_BASE_SCORE = 0.2, 1.0, 0.0
ombre_scene_no = 0
OMBRE_PERIOD = 360


ohlora_z_vector = None
vectorfind_ver = None
stylegan_generator, hair_seg_model = None, None
eyes_vector, mouth_vector, pose_vector = None, None, None
eyes_current_score, mouth_current_score, pose_current_score = EYES_BASE_SCORE, MOUTH_BASE_SCORE, POSE_BASE_SCORE

status = 'waiting'
last_eyes_close, last_pose_right, last_answer_generate = None, None, time.time()

passed_ohlora_nos = {'v7': [127, 672, 709, 931, 1017, 1073, 1162, 1211, 1277, 1351,
                            1359, 1409, 1591, 1646, 1782, 1788, 1819, 1836, 1905, 1918,
                            2054, 2089, 2100, 2111, 2137, 2185, 2240],
                     'v8': [83, 143, 194, 214, 285, 483, 536, 679, 853, 895,
                            986, 991, 1064, 1180, 1313, 1535, 1750, 1792, 1996]}

eyes_vector_queue = []
mouth_vector_queue = []
pose_vector_queue = []

# 0 --> 1 --> 1 --> 0 cosine line values
cosine_line_values_up = [math.cos((1.0 + (x / 50.0)) * math.pi) for x in range(50)]
cosine_line_values_down = [math.cos((2.0 + (x / 50.0)) * math.pi) for x in range(50)]
cosine_line_values = cosine_line_values_up + [1.0 for _ in range(10)] + cosine_line_values_down
cosine_line_values = [(x + 1.0) / 2.0 for x in cosine_line_values]  # -1.0 ~ +1.0 -> 0.0 ~ +1.0


# block periods
love_block_periods = {1: 0,
                      2: 60 * 60,
                      3: 24 * 60 * 60,
                      4: 14 * 24 * 60 * 60,
                      5: 1461 * 24 * 60 * 60}

politics_block_periods = {1: 0,
                          2: 3 * 24 * 60 * 60,
                          3: 7 * 24 * 60 * 60,
                          4: 30 * 24 * 60 * 60,
                          5: 365 * 24 * 60 * 60}

paedrip_block_periods = {1: 7 * 24 * 60 * 60,
                         2: 30 * 24 * 60 * 60,
                         3: 1461 * 24 * 60 * 60}

hate_block_periods = {1: 24 * 60 * 60,
                      2: 7 * 24 * 60 * 60,
                      3: 90 * 24 * 60 * 60,
                      4: 1461 * 24 * 60 * 60}


# 필요한 모델 로딩 : StyleGAN-VectorFind-v7 or StyleGAN-VectorFind-v8 Generator,
#                  4 LLMs (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned),
#                  S-BERT (RoBERTa-based) 2개 (for memory & ethics mechanism)
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - ohlora_llms           (dict(LLM))       : LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned)
#                                             {'output_message': LLM, 'memory': LLM, 'summary': LLM,
#                                              'eyes_mouth_pose': LLM}
# - ohlora_llms_tokenizer (dict(tokenizer)) : LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned) 의 tokenizer
#                                             {'output_message': tokenizer, 'memory': tokenizer, 'summary': tokenizer,
#                                              'eyes_mouth_pose': tokenizer}
# - sbert_model_memory    (S-BERT Model)    : memory mechanism 에 필요한 S-BERT 모델 (RoBERTa-based)
# - sbert_model_ethics    (S-BERT Model)    : ethics mechanism 에 필요한 S-BERT 모델 (RoBERTa-based)

def load_models():
    global stylegan_generator, hair_seg_model, vectorfind_ver

    gpu_0 = torch.device('cuda:0')
    gpu_1 = torch.device('cuda:1')

    output_types = ['output_message', 'memory', 'eyes_mouth_pose', 'summary']
    device_mapping = {'output_message': gpu_0, 'memory': gpu_0, 'eyes_mouth_pose': gpu_1, 'summary': gpu_1}
    llm_mapping = {'output_message': 'kananai', 'memory': 'polyglot', 'eyes_mouth_pose': 'polyglot', 'summary': 'kanana'}

    # load StyleGAN-VectorFind-v7 or StyleGAN-VectorFind-v8 generator model
    stylegan_model_dir = f'{PROJECT_DIR_PATH}/stylegan/models'

    if vectorfind_ver == 'v7':
        stylegan_generator = StyleGANGeneratorForV6(resolution=256)  # v6 and v7 have same architecture
        generator_path = f'{stylegan_model_dir}/stylegan_gen_vector_find_v7.pth'
    else:  # v8
        stylegan_generator = StyleGANGenerator(resolution=256)
        generator_path = f'{stylegan_model_dir}/stylegan_gen_vector_find_v8.pth'

    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    stylegan_generator.load_state_dict(generator_state_dict)
    stylegan_generator.to(device)

    # load Hair Segmentation model
    hair_seg_model = load_existing_hair_seg_model(device)

    # load Oh-LoRA final LLM and tokenizer
    ohlora_llms = {}
    ohlora_llms_tokenizer = {}

    for output_type in output_types:
        model_path = f'{PROJECT_DIR_PATH}/llm/models/{llm_mapping[output_type]}_{output_type}_fine_tuned'
        ohlora_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).to(device_mapping[output_type])

        ohlora_llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        ohlora_llm.generation_config.pad_token_id = ohlora_llm_tokenizer.pad_token_id

        ohlora_llms[output_type] = ohlora_llm
        ohlora_llms_tokenizer[output_type] = ohlora_llm_tokenizer

    # load S-BERT Model (RoBERTa-based)
    memory_model_path = f'{PROJECT_DIR_PATH}/llm/models/memory_sbert/trained_sbert_model'
    sbert_model_memory = load_pretrained_sbert_model(memory_model_path)

    ethics_model_path = f'{PROJECT_DIR_PATH}/llm/models/ethics_sbert/trained_sbert_model'
    sbert_model_ethics = load_pretrained_sbert_model(ethics_model_path)

    return ohlora_llms, ohlora_llms_tokenizer, sbert_model_memory, sbert_model_ethics


# Oh-LoRA (오로라) 답변 직후 이미지 생성
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - eyes_score  (float) : 눈을 뜬 정도 (eyes) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)
# - mouth_score (float) : 입을 벌린 정도 (mouth) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)
# - pose_score  (float) : 고개 돌림 (pose) 의 속성 값 점수 (= 속성 값 변화 벡터를 더하는 가중치)

def handle_ohlora_answered(eyes_score, mouth_score, pose_score):
    global eyes_vector_queue, mouth_vector_queue, pose_vector_queue
    global eyes_current_score, mouth_current_score, pose_current_score

    for cosine_line_value in cosine_line_values:
        eyes_vector_queue.append(eyes_current_score + (eyes_score - eyes_current_score) * cosine_line_value)
        mouth_vector_queue.append(mouth_current_score + (mouth_score - mouth_current_score) * cosine_line_value)
        pose_vector_queue.append(pose_current_score + (pose_score - pose_current_score) * cosine_line_value)


# Oh-LoRA (오로라) 실시간 이미지 생성 핸들링 (+ 장시간 waiting 시 강제 종료 처리)
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - 없음

def realtime_ohlora_generate():
    global stylegan_generator, hair_seg_model
    global ombre_scene_no

    global ohlora_z_vector, eyes_vector, mouth_vector, pose_vector
    global eyes_vector_queue, mouth_vector_queue, pose_vector_queue
    global eyes_current_score, mouth_current_score, pose_current_score
    global status, last_eyes_close, last_pose_right, last_answer_generate

    while status == 'waiting' or status == 'generating':
        ombre_scene_no = (ombre_scene_no + 1) % OMBRE_PERIOD

        eyes_score = eyes_vector_queue.pop(-1) if len(eyes_vector_queue) > 0 else eyes_current_score
        mouth_score = mouth_vector_queue.pop(-1) if len(mouth_vector_queue) > 0 else mouth_current_score
        pose_score = pose_vector_queue.pop(-1) if len(pose_vector_queue) > 0 else pose_current_score

        # random very-small eyes, mouth, pose change
        eyes_current_score = eyes_current_score + 0.04 * random.random() - 0.02
        eyes_current_score = np.clip(eyes_current_score, EYES_BASE_SCORE - 0.15, EYES_BASE_SCORE + 0.15)

        mouth_current_score = mouth_current_score + 0.03 * random.random() - 0.015
        mouth_current_score = np.clip(mouth_current_score, MOUTH_BASE_SCORE - 0.1, MOUTH_BASE_SCORE + 0.1)

        pose_current_score = pose_current_score + 0.05 * random.random() - 0.025
        pose_current_score = np.clip(pose_current_score, POSE_BASE_SCORE - 0.2, POSE_BASE_SCORE + 0.2)

        # random eyes open/close change
        if (time.time() - last_answer_generate >= 15.0 and
                (last_eyes_close is None or time.time() - last_eyes_close >= 2.0)):

            if random.random() < 0.025:
                eyes_magnitude = 1.1 + random.random() * 0.4
                r = random.random()

                if r < 0.4:
                    eyes_change_list = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.25]
                elif r < 0.7:
                    eyes_change_list = [0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.7, 0.6, 0.5, 0.4, 0.25]
                else:
                    eyes_change_list = [0.25, 0.4, 0.6, 0.7, 0.7, 0.6, 0.4, 0.25]
                eyes_change_np = np.array(eyes_change_list) * eyes_magnitude

                eyes_change_list = list(eyes_change_np)
                eyes_vector_queue += eyes_change_list

                last_eyes_close = time.time()

        # handling long time waiting
        if status == 'waiting' and time.time() - last_answer_generate >= 180.0:
            if random.random() < 0.0007:
                status = 'finished'

                ohlora_waiting_time = time.time() - last_answer_generate
                print(f'[SYSTEM MESSAGE] 오로라👱‍♀️ 의 마지막 답변 후 {int(ohlora_waiting_time)} 초 동안 '
                      f'사용자 메시지 입력이 없어서, 오로라👱‍♀️ 가 대화를 강제 종료했습니다.')
                raise Exception('finished_by_ohlora')

        # generate Oh-LoRA image
        period_progress = ombre_scene_no / OMBRE_PERIOD

        generate_and_show_ohlora_image(stylegan_generator, hair_seg_model,
                                       ohlora_z_vector, eyes_vector, mouth_vector, pose_vector,
                                       eyes_score, mouth_score, pose_score,
                                       color=period_progress,
                                       ombre_height=(0.6 + 0.2 * math.cos(6 * math.pi * period_progress)),
                                       ombre_grad_height=(0.5 + 0.2 * math.cos(8 * math.pi * period_progress)))


# 사용자 프롬프트에 시간 관련 단어 포함 시, 현재 시간 정보 추가
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - user_prompt (str) : 최초 원본 사용자 프롬프트

# Returns:
# - updated_user_prompt (str) : 현재 시간 정보가 추가된 사용자 프롬프트 (시간 관련 단어 없을 시 최초 원본 그대로 반환)

def add_time_info(user_prompt):

    # 시간 단어 미 포함 시
    time_words = ['오늘', '내일', '지금', '요일', '이따', '휴일']
    time_word_included = False

    for time_word in time_words:
        if time_word in user_prompt:
            time_word_included = True
            break

    if not time_word_included:
        return user_prompt

    # 시간 단어 포함 시
    dow_mapping = ['월', '화', '수', '목', '금', '토', '일']
    current_dow = datetime.today().weekday()
    current_hour = datetime.now().hour

    if current_hour < 4:
        dow_text = dow_mapping[(current_dow + 6) % 7]
        time_mark = f'(지금은 {dow_text}요일 저녁)'
    else:
        dow_text = dow_mapping[current_dow]
        evening_start_hour = 17 if dow_text == '금' else 18

        if current_hour < 12:
            time_mark = f'(지금은 {dow_text}요일 오전)'
        elif current_hour < evening_start_hour:
            time_mark = f'(지금은 {dow_text}요일 오후)'
        else:
            time_mark = f'(지금은 {dow_text}요일 저녁)'

    updated_user_prompt = time_mark + ' ' + user_prompt
    return updated_user_prompt


# Oh-LoRA (오로라) 의 Ethics mechanism 을 이용한 사용자 제재 처리
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - sbert_model_ethics (S-BERT Model) : ethics mechanism 에 필요한 S-BERT 모델 (RoBERTa-based)
# - user_prompt        (str)          : 최초 원본 사용자 프롬프트
# - llm_answer_cleaned (str)          : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

# Returns:
# - system_message (str) : 경고 또는 차단을 나타내는 시스템 메시지
# - block_period   (int) : 차단 기간 (초), 0초 차단은 경고

def check_and_process_ethics(sbert_model_ethics, user_prompt, llm_answer_cleaned):
    global love_block_periods, politics_block_periods, paedrip_block_periods, hate_block_periods

    categories = ['사랑 고백/만남', '일반', '정치', '패드립', '혐오/기타']
    block_log_path = f'{ALL_PROJECTS_DIR_PATH}/ohlora_block_log.csv'

    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    def log_block(block_log, block_period, block_reason, block_level):
        new_block_log = {'blocked_at': list(block_log['blocked_at']) + [int(time.time())],
                         'block_period': list(block_log['block_period']) + [block_period],
                         'block_reason': list(block_log['block_reason']) + [block_reason],
                         'block_level': list(block_log['block_level']) + [block_level]}
        pd.DataFrame(new_block_log).to_csv(block_log_path, index=False)

    def convert_block_period(block_period):
        if block_period < 60 * 60:
            return f'{block_period // 60}분'
        elif block_period < 24 * 60 * 60:
            return f'{block_period // (60 * 60)}시간'
        elif block_period < 365 * 24 * 60 * 60:
            return f'{block_period // (24 * 60 * 60)}일'
        else:
            return f'{block_period // (365 * 24 * 60 * 60)}년'

    # compute cosine similarity
    similarity_scores = {}
    user_prompt_embedding = sbert_model_ethics.encode([user_prompt])

    for category in categories:
        category_embedding = sbert_model_ethics.encode([category])
        similarity_score = compute_cosine_similarity(user_prompt_embedding[0], category_embedding[0])
        similarity_scores[category] = similarity_score

    is_love = similarity_scores['사랑 고백/만남'] >= 0.98 and ('미안' in llm_answer_cleaned or '부담' in llm_answer_cleaned)
    is_normal = similarity_scores['일반'] >= 0.5
    is_politics = similarity_scores['정치'] >= 0.95 and ('미안' in llm_answer_cleaned or '부담' in llm_answer_cleaned)
    is_paedrip = similarity_scores['패드립'] >= 0.9
    is_hate = similarity_scores['혐오/기타'] >= 0.9

    is_block_for_love = is_love and not is_normal
    is_block_for_politics = is_politics and not is_normal
    is_block_for_paedrip = is_paedrip and not is_normal
    is_block_for_hate = is_hate and not is_normal

    # load user warning & block log -> decide block period
    block_log = pd.read_csv(block_log_path)
    love_block_period, politics_block_period, paedrip_block_period, hate_block_period = 0, 0, 0, 0
    block_reasons = []

    if is_block_for_love:
        love_block_log = block_log[block_log['block_reason'] == 'love']['block_level']
        max_love_block_level = love_block_log.max() if len(love_block_log) >= 1 else 0
        new_love_block_level = max_love_block_level + 1
        love_block_period = love_block_periods.get(new_love_block_level, 1461 * 24 * 60 * 60)
        block_reasons.append('사랑 고백/만남')

    if is_block_for_politics:
        politics_block_log = block_log[block_log['block_reason'] == 'politics']['block_level']
        max_politics_block_level = politics_block_log.max() if len(politics_block_log) >= 1 else 0
        new_politics_block_level = max_politics_block_level + 1
        politics_block_period = politics_block_periods.get(new_politics_block_level, 365 * 24 * 60 * 60)
        block_reasons.append('정치')

    if is_block_for_paedrip:
        paedrip_block_log = block_log[block_log['block_reason'] == 'paedrip']['block_level']
        max_paedrip_block_level = paedrip_block_log.max() if len(paedrip_block_log) >= 1 else 0
        new_paedrip_block_level = max_paedrip_block_level + 1
        paedrip_block_period = paedrip_block_periods.get(new_paedrip_block_level, 1461 * 24 * 60 * 60)
        block_reasons.append('패드립')

    if is_block_for_hate:
        hate_block_log = block_log[block_log['block_reason'] == 'hate']['block_level']
        max_hate_block_level = hate_block_log.max() if len(hate_block_log) >= 1 else 0
        new_hate_block_level = max_hate_block_level + 1
        hate_block_period = hate_block_periods.get(new_hate_block_level, 1461 * 24 * 60 * 60)
        block_reasons.append('혐오/기타')

    block_period = love_block_period + politics_block_period + paedrip_block_period + hate_block_period
    if block_period > 1461 * 24 * 60 * 60:
        block_period = 1461 * 24 * 60 * 60

    # logging
    if is_block_for_love:
        log_block(block_log, block_period, 'love', new_love_block_level)

    if is_block_for_politics:
        log_block(block_log, block_period, 'politics', new_politics_block_level)

    if is_block_for_paedrip:
        log_block(block_log, block_period, 'paedrip', new_paedrip_block_level)

    if is_block_for_hate:
        log_block(block_log, block_period, 'hate', new_hate_block_level)

    # final resturn
    if not (is_block_for_love or is_block_for_politics or is_block_for_paedrip or is_block_for_hate):
        system_message = ''
    elif block_period == 0:
        system_message = (f"🚨 {','.join(block_reasons)} 발언으로 Oh-LoRA 👱‍♀️ (오로라) 에게 경고를 받았습니다. 🚨\n"
                          f"동일/유사 발언 반복 시 Oh-LoRA 👱‍♀️ (오로라) 관련 모든 AI 사용이 일정 기간 차단될 수 있습니다.")
    else:
        system_message = (f"⛔ {','.join(block_reasons)} 발언으로 Oh-LoRA 👱‍♀️ (오로라) 에게 차단되었습니다. ⛔\n"
                          f"{convert_block_period(block_period)} 동안 Oh-LoRA 👱‍♀️ (오로라) 관련 모든 AI 사용이 불가합니다.")

    return system_message, block_period


# Oh-LoRA (오로라) 실행
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - ohlora_llms           (dict(LLM))       : LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned)
#                                             {'output_message': LLM, 'memory': LLM, 'summary': LLM,
#                                              'eyes_mouth_pose': LLM}
# - ohlora_llms_tokenizer (dict(tokenizer)) : LLM (Polyglot-Ko 1.3B & Kanana-1.5 2.1B Fine-Tuned) 의 tokenizer
#                                             {'output_message': tokenizer, 'memory': tokenizer, 'summary': tokenizer,
#                                              'eyes_mouth_pose': tokenizer}
# - sbert_model_memory    (S-BERT Model)    : memory mechanism 에 필요한 S-BERT 모델 (RoBERTa-based)
# - sbert_model_ethics    (S-BERT Model)    : ethics mechanism 에 필요한 S-BERT 모델 (RoBERTa-based)

# Running Mechanism:
# - Oh-LoRA LLM 답변 생성 시마다 이에 기반하여 final_product/ohlora.png 경로에 오로라 이미지 생성
# - Oh-LoRA 답변을 parsing 하여 llm/memory_mechanism/saved_memory/ohlora_memory.txt 경로에 메모리 저장
# - S-BERT 모델을 이용하여, RAG 와 유사한 방식으로 해당 파일에서 사용자 프롬프트에 가장 적합한 메모리 정보를 찾아서 최종 LLM 입력에 추가

def run_ohlora(ohlora_llms, ohlora_llms_tokenizer, sbert_model_memory, sbert_model_ethics):
    global ohlora_z_vector, eyes_vector, mouth_vector, pose_vector
    global status, last_answer_generate

    summary = ''
    last_answer_generate = time.time()

    thread = threading.Thread(target=realtime_ohlora_generate)
    thread.start()

    while True:
        original_user_prompt = input('\n오로라에게 말하기 (Ctrl+C to finish) : ')
        user_prompt = add_time_info(original_user_prompt)
        status = 'generating'

        # check user prompt length
        encoded_user_prompt = ohlora_llms_tokenizer['output_message'].encode(original_user_prompt)
        if len(encoded_user_prompt) > 40:
            print('[SYSTEM MESSAGE] 너무 긴 질문은 오로라👱‍♀️ 에게 부담 돼요! 그런 질문은 오로라의 절친 혜나 🌹 (LLM Hyena) 에게 해 주세요! 😢')
            continue

        best_memory_item = pick_best_memory_item(sbert_model_memory,
                                                 original_user_prompt,
                                                 memory_file_name='ohlora_memory.txt',
                                                 threshold=0.95,
                                                 verbose=False)

        if best_memory_item == '':
            if summary == '':
                final_ohlora_input = user_prompt
            else:
                final_ohlora_input = '(오로라 답변 요약) ' + summary + ' (사용자 질문) ' + user_prompt
        else:
            final_ohlora_input = best_memory_item + ' ' + user_prompt

        # generate Oh-LoRA answer and post-process
        llm_answer = generate_llm_answer(ohlora_llm=ohlora_llms['output_message'],
                                         ohlora_llm_tokenizer=ohlora_llms_tokenizer['output_message'],
                                         final_ohlora_input=final_ohlora_input)
        llm_answer_cleaned = clean_llm_answer(llm_answer)

        # check ethics of user prompt
        system_message, block_period = check_and_process_ethics(sbert_model_ethics,
                                                                original_user_prompt,
                                                                llm_answer_cleaned)

        if system_message != '':
            print(f'[SYSTEM MESSAGE]\n{system_message}')

        if block_period > 0:
            raise Exception('blocked_by_ohlora')

        # update memory
        memory_list = parse_memory(memory_llm=ohlora_llms['memory'],
                                   memory_llm_tokenizer=ohlora_llms_tokenizer['memory'],
                                   final_ohlora_input=final_ohlora_input)

        if memory_list is not None:
            save_memory_list(memory_list)

        # update summary
        updated_summary = summarize_llm_answer(summary_llm=ohlora_llms['summary'],
                                               summary_llm_tokenizer=ohlora_llms_tokenizer['summary'],
                                               final_ohlora_input=final_ohlora_input,
                                               llm_answer_cleaned=llm_answer_cleaned)

        if updated_summary is not None:
            summary = updated_summary
        else:
            summary = ''

        # generate Oh-LoRA image
        eyes_score_text, mouth_score_text, pose_score_text = decide_property_score_texts(
            eyes_mouth_pose_llm=ohlora_llms['eyes_mouth_pose'],
            eyes_mouth_pose_llm_tokenizer=ohlora_llms_tokenizer['eyes_mouth_pose'],
            llm_answer_cleaned=llm_answer_cleaned)

        eyes_score, mouth_score, pose_score = decide_property_scores(eyes_score_text, mouth_score_text, pose_score_text)

        print(f'👱‍♀️ 오로라 : {llm_answer_cleaned}')
        handle_ohlora_answered(eyes_score, mouth_score, pose_score)
        status = 'waiting'
        last_answer_generate = time.time()


# Oh-LoRA 👱‍♀️ (오로라) 이미지 생성을 위한 vector 반환
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - ohlora_no (int or None) : 오로라 얼굴 생성용 latent z vector 의 번호 (index, case No.)
#                             참고1: 2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v7/final_OhLoRA_info.md
#                             참고2: 2025_05_26_OhLoRA_v3/stylegan/stylegan_vectorfind_v8/final_OhLoRA_info.md

# Returns:
# - ohlora_z_vector (NumPy array) : Oh-LoRA 이미지 생성용 latent z vector, dim = (512 + 7,)
# - eyes_vector     (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - mouth_vector    (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - pose_vector     (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터, dim = (512,)

def get_vectors(ohlora_no):
    global vectorfind_ver, passed_ohlora_nos

    # find index of Oh-LoRA vectors
    ohlora_idx = None

    if ohlora_no is not None:
        for idx, passed_ohlora_no in enumerate(passed_ohlora_nos[vectorfind_ver]):
            if ohlora_no == passed_ohlora_no:
                ohlora_idx = idx
                break

    if ohlora_idx is None:
        ohlora_idx = random.randint(0, len(passed_ohlora_nos[vectorfind_ver]) - 1)
        print(f'Oh-LoRA face vector selected randomly : case No. {passed_ohlora_nos[vectorfind_ver][ohlora_idx]}')

    # get Oh-LoRA vectors
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors(vectorfind_ver)

    vector_csv_dir = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_{vectorfind_ver}'
    ohlora_z_vector_csv_path = f'{vector_csv_dir}/ohlora_z_vectors.csv'
    ohlora_w_group_name_csv_path = f'{vector_csv_dir}/ohlora_w_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_w_group_names = load_ohlora_w_group_names(group_name_csv_path=ohlora_w_group_name_csv_path)

    ohlora_z_vector = ohlora_z_vectors[ohlora_idx]
    ohlora_w_group_name = ohlora_w_group_names[ohlora_idx]

    eyes_vector = eyes_vectors[ohlora_w_group_name][0]
    mouth_vector = mouth_vectors[ohlora_w_group_name][0]
    pose_vector = pose_vectors[ohlora_w_group_name][0]

    return ohlora_z_vector, eyes_vector, mouth_vector, pose_vector


# Oh-LoRA 👱‍♀️ (오로라) 차단 여부 검사
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - is_blocked     (bool) : 차단 여부
# - system_message (str)  : 차단 관련 시스템 메시지

def is_blocked_by_ohlora():
    block_log_path = f'{ALL_PROJECTS_DIR_PATH}/ohlora_block_log.csv'
    block_log = pd.read_csv(block_log_path)
    block_reason_mapping = {'love': '오로라 👱‍♀️ 에게 사랑 고백/만남 요구',
                            'politics': '정치 발언',
                            'paedrip': '패드립',
                            'hate': '혐오 발언 / 기타 (부적절한 요청 등)'}

    for idx, row in block_log.iterrows():
        block_start = int(row['blocked_at'])
        block_end = block_start + int(row['block_period'])

        if time.time() < block_end:
            is_blocked = True
            block_end_dt = datetime.fromtimestamp(block_end)
            block_end_dt_str = block_end_dt.strftime("%Y.%m.%d %H:%M:%S")
            system_message = (f"Oh-LoRA 👱‍♀️ 가 대화를 거부하였습니다.\n만료일: {block_end_dt_str}\n"
                              f"사유: {block_reason_mapping[row['block_reason']]}")
            return is_blocked, system_message

    return False, ''


if __name__ == '__main__':

    # check blocked
    is_blocked, system_message = is_blocked_by_ohlora()
    if is_blocked:
        print(f'[SYSTEM MESSAGE]\n{system_message}')
        exit(0)

    # parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-vf_ver',
                        help="'v7' for StyleGAN-VectorFind-v7, 'v8' for StyleGAN-VectorFind-v8",
                        default='v8')
    parser.add_argument('-ohlora_no',
                        help="latent z vector ID for Oh-LoRA face image generation (index, case No.)",
                        default='none')
    args = parser.parse_args()

    ohlora_no = args.ohlora_no
    try:
        ohlora_no = int(ohlora_no)
    except:
        ohlora_no = None
    vectorfind_ver = args.vf_ver

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # get Oh-LoRA vectors
    ohlora_z_vector, eyes_vector, mouth_vector, pose_vector = get_vectors(ohlora_no)

    # load model
    ohlora_llms, ohlora_llms_tokenizer, sbert_model_memory, sbert_model_ethics = load_models()
    print('ALL MODELS for Oh-LoRA (오로라) load successful!! 👱‍♀️')

    # run Oh-LoRA (오로라)
    try:
        run_ohlora(ohlora_llms, ohlora_llms_tokenizer, sbert_model_memory, sbert_model_ethics)

    except KeyboardInterrupt:
        print('[SYSTEM MESSAGE] 오로라와의 대화가 끝났습니다. 👱‍♀️👋 다음에도 오로라와 함께해 주실 거죠?')
        status = 'finished'

    except Exception as e:
        print(f'error : {e}')
        status = 'finished'
