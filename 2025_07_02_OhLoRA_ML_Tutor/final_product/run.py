
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import math
import numpy as np
import pandas as pd
import random

import argparse
import threading
import os
import sys
import time
from datetime import datetime

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ALL_PROJECTS_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(PROJECT_DIR_PATH)

from run_display import generate_and_show_ohlora_image
from run_llm import generate_llm_answer, clean_llm_answer

from ai_qna.run_rag_concept import pick_best_db_item_csv
from ai_quiz.select_quiz.select_quiz import select_next_quiz
from ai_quiz.sbert.inference_sbert import run_inference_each_example
from ai_interview.rag_concept_common import pick_best_candidate

from stylegan.stylegan_common.stylegan_generator import StyleGANGenerator, StyleGANGeneratorForV6
from stylegan.stylegan_vectorfind import (load_ohlora_z_vectors,
                                          load_ohlora_w_group_names,
                                          get_property_change_vectors)
from ombre.load_seg_model import load_existing_hair_seg_model


EYES_BASE_SCORE, MOUTH_BASE_SCORE, POSE_BASE_SCORE = 0.2, 1.0, 0.0
ombre_scene_no = 0
OMBRE_PERIOD = 360
QUIZ_LIST_PATH = f'{PROJECT_DIR_PATH}/ai_quiz/dataset/question_list.csv'


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


# Pre-trained (or Fine-Tuned) S-BERT Model 로딩
# Reference : https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-라이브러리를-활용한-SBERT-학습-방법
# Create Date : 2025.08.01
# Last Update Date : -

# Arguments:
# - model_path (str) : Pre-trained (or Fine-Tuned) S-BERT Model 의 경로

# Returns:
# - pretrained_sbert_model (S-BERT Model) : Pre-train 된 Sentence-BERT 모델

def load_pretrained_sbert_model(model_path):
    embedding_model = models.Transformer(
        model_name_or_path=model_path,
        max_seq_length=64,
        do_lower_case=True
    )

    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    pretrained_sbert_model = SentenceTransformer(modules=[embedding_model, pooling_model])
    return pretrained_sbert_model


# function type 에 따라 필요한 LLM & S-BERT 모델 로딩
# Create Date : 2025.08.10
# Last Update Date : -

# Arguments:
# - function_type (str)    : 실행할 기능으로, 'qna', 'quiz', 'interview' 중 하나
# - gpu_0         (device) : 1번째 GPU
# - gpu_1         (device) : 2번째 GPU

# Returns:
# - model_dict (dict) : LLM & S-BERT Model 반환용 dictionary (key: 'llm', 'sbert', 'llm_tokenizer' 등)

def load_llm_and_sbert_model(function_type, gpu_0, gpu_1):
    assert function_type in ['qna', 'quiz', 'interview']
    model_dict = {}

    # LLM & S-BERT model path
    if function_type == 'qna':
        llm_path = f'{PROJECT_DIR_PATH}/ai_qna/models/kananai_sft_final_fine_tuned'
        sbert_path = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/trained_sbert_model'

    elif function_type == 'quiz':
        llm_path = f'{PROJECT_DIR_PATH}/ai_quiz/models/kananai_sft_final_fine_tuned_10epochs'
        sbert_path = f'{PROJECT_DIR_PATH}/ai_quiz/models/sbert/trained_sbert_model'

    else:  # interview
        llm_path = f'{PROJECT_DIR_PATH}/ai_interview/models/kananai_sft_final_fine_tuned_5epochs'
        sbert_path_next_question = f'{PROJECT_DIR_PATH}/ai_interview/models/next_question_sbert/trained_sbert_model_40'
        sbert_path_output_answer = f'{PROJECT_DIR_PATH}/ai_interview/models/output_answer_sbert/trained_sbert_model_40'

    # load LLM and tokenizer
    model_dict['llm'] = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(gpu_0)
    model_dict['llm_tokenizer'] = AutoTokenizer.from_pretrained(llm_path)
    model_dict['llm'].generation_config.pad_token_id = model_dict['llm_tokenizer'].pad_token_id

    if function_type == 'qna' or function_type == 'quiz':
        model_dict['sbert'] = load_pretrained_sbert_model(sbert_path)

    else:
        model_dict['sbert_next_question'] = load_pretrained_sbert_model(sbert_path_next_question)
        model_dict['sbert_output_answer'] = load_pretrained_sbert_model(sbert_path_output_answer)

    return model_dict


# 필요한 모델 로딩 : StyleGAN-VectorFind-v7 or StyleGAN-VectorFind-v8 Generator + LLM, S-BERT models
# Create Date : 2025.08.10
# Last Update Date : -

# Arguments:
# - function_type (str) : 실행할 기능으로, 'qna', 'quiz', 'interview' 중 하나

# Returns:
# - sbert_model_ethics (S-BERT Model) : ethics mechanism 에 필요한 S-BERT 모델 (RoBERTa-based)
# - model_dict         (dict)         : LLM & S-BERT Model 저장용 dictionary (key: 'llm', 'sbert', 'llm_tokenizer' 등)

def load_models(function_type):
    global stylegan_generator, hair_seg_model, vectorfind_ver

    gpu_0 = torch.device('cuda:0')
    gpu_1 = torch.device('cuda:1')

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

    # load LLM and S-BERT models according to function type
    model_dict = load_llm_and_sbert_model(function_type, gpu_0, gpu_1)
    print(f'LLM and S-BERT models load successful!! (function type: {function_type})  👱‍♀️✨')

    # load Hair Segmentation model
    hair_seg_model = load_existing_hair_seg_model(device)

    # load S-BERT Model (RoBERTa-based)
    ethics_model_path = f'{PROJECT_DIR_PATH}/final_product/models/ethics_sbert/trained_sbert_model'
    sbert_model_ethics = load_pretrained_sbert_model(ethics_model_path)

    return sbert_model_ethics, model_dict


# Oh-LoRA (오로라) 답변 직후 이미지 생성
# Create Date : 2025.08.01
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
# Create Date : 2025.08.01
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


# Oh-LoRA (오로라) 의 Ethics mechanism 을 이용한 사용자 제재 처리
# Create Date : 2025.08.01
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


# Oh-LoRA (오로라) 실행 중 'qna' (머신러닝 Q&A) 기능 처리
# Create Date : 2025.08.10
# Last Update Date : -

# Arguments:
# - user_prompt (str)  : 최초 원본 사용자 프롬프트
# - model_dict  (dict) : LLM & S-BERT Model 저장용 dictionary

# Returns:
# - llm_answer (str) : Oh-LoRA LLM 최종 답변

def run_ohlora_qna(user_prompt, model_dict):
    best_db_item = pick_best_db_item_csv(sbert_model=model_dict['sbert'],
                                         user_prompt=user_prompt,
                                         db_file_name='rag_data_text.csv',
                                         verbose=False)
    final_llm_prompt = f'(제공된 정보) {best_db_item} (사용자 질문) {user_prompt}'

    llm_answer = generate_llm_answer(ohlora_llm=model_dict['llm'],
                                     ohlora_llm_tokenizer=model_dict['llm_tokenizer'],
                                     final_ohlora_input=final_llm_prompt,
                                     function_type='qna')

    return llm_answer


# Oh-LoRA (오로라) 실행 중 'quiz' (머신러닝 퀴즈) 기능 처리
# Create Date : 2025.09.23
# Last Update Date : -

# Arguments:
# - quiz_current_quiz_info (dict) : 현재 퀴즈 정보 (keys: ['idx', 'quiz', 'keywords', 'good_answer'])
# - user_prompt            (str)  : 최초 원본 사용자 프롬프트 (= 퀴즈 답변)
# - model_dict             (dict) : LLM & S-BERT Model 저장용 dictionary

# Returns:
# - llm_answer     (str)   : Oh-LoRA LLM 최종 답변
# - rounded_score  (float) : 사용자 답변 채점 결과 점수 (0.0 - 1.0 범위)
# - next_quiz_info (dict)  : 다음 퀴즈 정보 (keys 구성은 quiz_current_quiz_info 와 동일)

def run_ohlora_quiz(quiz_current_quiz_info, user_prompt, model_dict):
    quiz_log_path = f'{PROJECT_DIR_PATH}/ai_quiz/select_quiz/quiz_log.csv'
    quiz_list_df = pd.read_csv(QUIZ_LIST_PATH)

    # generate answer
    quiz = quiz_current_quiz_info['quiz']
    quiz_idx = quiz_current_quiz_info['idx']
    keywords = quiz_current_quiz_info['keywords']
    good_answer = quiz_current_quiz_info['good_answer']

    final_llm_prompt = f'(퀴즈 문제) {quiz} (모범 답안) {good_answer} (사용자 답변) {user_prompt}'

    llm_answer = generate_llm_answer(ohlora_llm=model_dict['llm'],
                                     ohlora_llm_tokenizer=model_dict['llm_tokenizer'],
                                     final_ohlora_input=final_llm_prompt,
                                     function_type='quiz')

    predicted_score = run_inference_each_example(model_dict['sbert'], user_prompt, good_answer)

    # save score
    rounded_score = round(np.clip(round(20.0 * predicted_score) / 20.0, 0.0, 1.0), 2)

    if os.path.exists(quiz_log_path):
        quiz_log_df = pd.read_csv(quiz_log_path)
        quiz_log_df = quiz_log_df[quiz_log_df['quiz_no'] != quiz_idx]
        quiz_log_df.loc[len(quiz_log_df)] = {'quiz_no': quiz_idx, 'keywords': keywords, 'score': rounded_score}

    else:
        quiz_log_dict = {'quiz_no': [quiz_idx], 'keywords': [keywords], 'score': [rounded_score]}
        quiz_log_df = pd.DataFrame(quiz_log_dict)

    quiz_log_df.to_csv(quiz_log_path, index=False)

    # select next quiz
    next_quiz_idx = select_next_quiz(quiz_log_path)
    next_quiz_info = {
        'idx': next_quiz_idx,
        'quiz': quiz_list_df['quiz'][next_quiz_idx],
        'keywords': quiz_list_df['keywords'][next_quiz_idx],
        'good_answer': quiz_list_df['good_answer'][next_quiz_idx]
    }

    return llm_answer, rounded_score, next_quiz_info


# Oh-LoRA (오로라) 실행 중 'interview' (머신러닝 분야 모의 인터뷰) 기능 처리
# Create Date : 2025.09.23
# Last Update Date : -

# Arguments:
# - current_question (str)         : LLM이 생성할 질문의 주제
# - user_prompt      (str or None) : 최초 원본 사용자 프롬프트 (질문에 대한 사용자 답변)
# - model_dict       (dict)        : LLM & S-BERT Model 저장용 dictionary

# Returns:
# - llm_answer    (str) : Oh-LoRA LLM 최종 답변
# - next_question (str) : LLM이 다음에 생성할 질문의 주제

def run_ohlora_interview(current_question, user_prompt, model_dict):

    # 면접 시작 인사
    if user_prompt is None:
        llm_answer = generate_llm_answer(ohlora_llm=model_dict['llm'],
                                         ohlora_llm_tokenizer=model_dict['llm_tokenizer'],
                                         final_ohlora_input='면접 시작',
                                         function_type='interview')
        next_question = '면접 시작 인사'

    # 질의응답
    else:
        sbert_model_output_answer = model_dict['sbert_output_answer']
        best_candidate_info = pick_best_candidate(sbert_model=sbert_model_output_answer,
                                                  user_prompt=user_prompt,
                                                  candidates_csv_name='embeddings_answer_type.csv',
                                                  verbose=True)

        print(best_candidate_info)

    print('llm_answer :', llm_answer)
    print('next_question :', next_question)

    return llm_answer, next_question


# Oh-LoRA (오로라) 실행
# Create Date : 2025.09.23
# Last Update Date : -

# Arguments:
# - function_type      (str)          : 실행할 기능으로, 'qna', 'quiz', 'interview' 중 하나
# - model_dict         (dict)         : LLM & S-BERT Model 저장용 dictionary (key: 'llm', 'sbert' 등)
# - sbert_model_ethics (S-BERT Model) : ethics mechanism 에 필요한 S-BERT 모델 (RoBERTa-based)

# Running Mechanism:
# - Oh-LoRA LLM 답변 생성 시마다 이에 기반하여 final_product/ohlora.png 경로에 오로라 이미지 생성
# - Oh-LoRA 답변을 parsing 하여 llm/memory_mechanism/saved_memory/ohlora_memory.txt 경로에 메모리 저장
# - S-BERT 모델을 이용하여, RAG 와 유사한 방식으로 해당 파일에서 사용자 프롬프트에 가장 적합한 메모리 정보를 찾아서 최종 LLM 입력에 추가

def run_ohlora(function_type, model_dict, sbert_model_ethics):
    global ohlora_z_vector, eyes_vector, mouth_vector, pose_vector
    global status, last_answer_generate
    last_answer_generate = time.time()

    interview_current_question = ''
    quiz_current_quiz_info = {'quiz': '', 'good_answer': ''}

    thread = threading.Thread(target=realtime_ohlora_generate)
    thread.start()

    if function_type == 'qna':
        user_prompt_prefix = '오로라에게 머신러닝 질문하기'
        stop_sequence = '(답변 종료'

    elif function_type == 'quiz':
        user_prompt_prefix = '오로라의 퀴즈에 답하기'
        stop_sequence = '(해설 종료'

        quiz_list_csv = pd.read_csv(QUIZ_LIST_PATH, index_col=0)
        quiz_current_quiz = quiz_list_csv[['quiz', 'good_answer', 'keywords']].sample(n=1)

        quiz_current_quiz_info['idx'] = quiz_current_quiz.index.values[0]
        quiz_current_quiz_info['quiz'] = quiz_current_quiz['quiz'].item()
        quiz_current_quiz_info['good_answer'] = quiz_current_quiz['good_answer'].item()
        quiz_current_quiz_info['keywords'] = quiz_current_quiz['keywords'].item()

        print(f"\n[ QUIZ 🙋‍♀️ ]\n{quiz_current_quiz_info['quiz']}")

    else:  # interview
        user_prompt_prefix = '오로라의 면접 질문에 답하기'
        stop_sequence = '(발화 종료'

        first_greeting, _ = run_ohlora_interview(current_question='', user_prompt=None, model_dict=model_dict)
        interview_current_question = '면접 시작 인사'
        print(f"\n👱‍♀️ 오로라 : {first_greeting.replace(stop_sequence, '')}")

    while True:
        original_user_prompt = input(f'\n{user_prompt_prefix} (Ctrl+C to finish) : ')

        # function type ('qna', 'quiz' or 'interview') 에 따른 처리
        if function_type == 'qna':
            llm_answer = run_ohlora_qna(original_user_prompt, model_dict)

        elif function_type == 'quiz':
            llm_answer, user_score, next_quiz_info = (
                run_ohlora_quiz(quiz_current_quiz_info, original_user_prompt, model_dict))

            last_question_good_answer = quiz_current_quiz_info['good_answer']
            quiz_current_quiz_info = next_quiz_info

        else:  # interview
            llm_answer, next_question = (
                run_ohlora_interview(interview_current_question, original_user_prompt, model_dict))
            interview_current_question = next_question

        llm_answer_cleaned = clean_llm_answer(llm_answer)

        # check ethics of user prompt
        system_message, block_period = check_and_process_ethics(sbert_model_ethics,
                                                                original_user_prompt,
                                                                llm_answer_cleaned)

        if system_message != '':
            print(f'[SYSTEM MESSAGE]\n{system_message}')

        if block_period > 0:
            raise Exception('blocked_by_ohlora')

        # generate Oh-LoRA image
        eyes_score, mouth_score, pose_score = 0.0, 0.0, 0.0  # TODO: temp

        print(f"👱‍♀️ 오로라 : {llm_answer_cleaned.replace(stop_sequence, '')}")
        if function_type == 'quiz':
            print(f'👱‍♀️ 오로라의 채점 결과 : {round(100 * user_score)} 점')
            print(f'👍 모범 답안 : {last_question_good_answer}')

        handle_ohlora_answered(eyes_score, mouth_score, pose_score)
        status = 'waiting'
        last_answer_generate = time.time()

        # print next quiz / interview question
        if function_type == 'quiz':
            print(f"\n[ QUIZ 🙋‍♀️ ]\n{quiz_current_quiz_info['quiz']}")
        elif function_type == 'interview':
            pass  # TODO implement


# Oh-LoRA 👱‍♀️ (오로라) 이미지 생성을 위한 vector 반환
# Create Date : 2025.08.01
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
# Create Date : 2025.08.01
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
    parser.add_argument('-function_type',
                        help="function type (one of 'qna', 'quiz' and 'interview')",
                        default='qna')
    args = parser.parse_args()

    ohlora_no = args.ohlora_no
    try:
        ohlora_no = int(ohlora_no)
    except:
        ohlora_no = None
    vectorfind_ver = args.vf_ver
    function_type = args.function_type

    assert function_type in ['qna', 'quiz', 'interview']

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # get Oh-LoRA vectors
    ohlora_z_vector, eyes_vector, mouth_vector, pose_vector = get_vectors(ohlora_no)

    # load model
    sbert_model_ethics, model_dict = load_models(function_type)
    print('ALL MODELS for Oh-LoRA (오로라) load successful!! 👱‍♀️')

    # run Oh-LoRA (오로라)
    try:
        run_ohlora(function_type, model_dict, sbert_model_ethics)

    except KeyboardInterrupt:
        print('[SYSTEM MESSAGE] 오로라와의 대화가 끝났습니다. 👱‍♀️👋 다음에도 오로라와 함께해 주실 거죠?')
        status = 'finished'

    except Exception as e:
        print(f'error : {e}')
        status = 'finished'
