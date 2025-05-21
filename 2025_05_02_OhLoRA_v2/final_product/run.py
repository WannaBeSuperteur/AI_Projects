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
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from run_llm import (generate_llm_answer, clean_llm_answer, parse_memory, save_memory_list, summarize_llm_answer,
                     decide_property_score_texts, decide_property_scores)
from run_display import generate_and_show_ohlora_image

from stylegan.stylegan_common.stylegan_generator import StyleGANGeneratorForV6
from stylegan.run_stylegan_vectorfind_v7 import (load_ohlora_z_vectors,
                                                 load_ohlora_w_group_names,
                                                 get_property_change_vectors)

from llm.memory_mechanism.load_sbert_model import load_pretrained_sbert_model
from llm.run_memory_mechanism import pick_best_memory_item


EYES_BASE_SCORE, MOUTH_BASE_SCORE, POSE_BASE_SCORE = 0.2, 1.0, 0.0


ohlora_z_vector = None
eyes_vector, mouth_vector, pose_vector = None, None, None
eyes_current_score, mouth_current_score, pose_current_score = EYES_BASE_SCORE, MOUTH_BASE_SCORE, POSE_BASE_SCORE

status = 'waiting'
last_eyes_close, last_pose_right, last_answer_generate = None, None, time.time()

passed_ohlora_nos = [127, 672, 709, 931, 1017, 1073, 1162, 1211, 1277, 1351,
                     1359, 1409, 1591, 1646, 1782, 1788, 1819, 1836, 1905, 1918,
                     2054, 2089, 2100, 2111, 2137, 2185, 2240]

eyes_vector_queue = []
mouth_vector_queue = []
pose_vector_queue = []

# 0 --> 1 --> 1 --> 0 cosine line values
cosine_line_values_up = [math.cos((1.0 + (x / 30.0)) * math.pi) for x in range(30)]
cosine_line_values_down = [math.cos((2.0 + (x / 30.0)) * math.pi) for x in range(30)]
cosine_line_values = cosine_line_values_up + [1.0 for _ in range(10)] + cosine_line_values_down
cosine_line_values = [(x + 1.0) / 2.0 for x in cosine_line_values]


# 필요한 모델 로딩 : StyleGAN-VectorFind-v7 Generator, 4 LLMs (Polyglot-Ko 1.3B Fine-Tuned), S-BERT (RoBERTa-based)
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - stylegan_generator    (nn.Module)       : StyleGAN-VectorFind-v7 generator
# - ohlora_llms           (dict(LLM))       : LLM (Polyglot-Ko 1.3B Fine-Tuned)
#                                             {'output_message': LLM, 'memory': LLM, 'summary': LLM,
#                                              'eyes_mouth_pose': LLM}
# - ohlora_llms_tokenizer (dict(tokenizer)) : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
#                                             {'output_message': tokenizer, 'memory': tokenizer, 'summary': tokenizer,
#                                              'eyes_mouth_pose': tokenizer}
# - sbert_model           (S-BERT Model)    : S-BERT (RoBERTa-based)

def load_models():
    gpu_0 = torch.device('cuda:0')
    gpu_1 = torch.device('cuda:1')

    output_types = ['output_message', 'memory', 'eyes_mouth_pose', 'summary']
    device_mapping = {'output_message': gpu_0, 'memory': gpu_0, 'eyes_mouth_pose': gpu_1, 'summary': gpu_1}

    # load StyleGAN-VectorFind-v7 generator model
    stylegan_generator = StyleGANGeneratorForV6(resolution=256)  # v6 and v7 have same architecture
    stylegan_model_dir = f'{PROJECT_DIR_PATH}/stylegan/models'
    generator_path = f'{stylegan_model_dir}/stylegan_gen_vector_find_v7.pth'

    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    stylegan_generator.load_state_dict(generator_state_dict)
    stylegan_generator.to(device)

    # load Oh-LoRA final LLM and tokenizer
    ohlora_llms = {}
    ohlora_llms_tokenizer = {}

    for output_type in output_types:
        model_path = f'{PROJECT_DIR_PATH}/llm/models/polyglot_{output_type}_fine_tuned'
        ohlora_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).to(device_mapping[output_type])

        ohlora_llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        ohlora_llm.generation_config.pad_token_id = ohlora_llm_tokenizer.pad_token_id

        ohlora_llms[output_type] = ohlora_llm
        ohlora_llms_tokenizer[output_type] = ohlora_llm_tokenizer

    # load S-BERT Model (RoBERTa-based)
    model_path = f'{PROJECT_DIR_PATH}/llm/models/memory_sbert/trained_sbert_model'
    sbert_model = load_pretrained_sbert_model(model_path)

    return stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model


# Oh-LoRA (오로라) 답변 직후 이미지 생성
# Create Date : 2025.05.20
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
# Create Date : 2025.05.20
# Last Update Date : 2025.05.21
# - 사용자 입력 없을 시 오로라가 대화를 강제 종료하는 조건 수정

# Arguments:
# - 없음

def realtime_ohlora_generate():
    global ohlora_z_vector, eyes_vector, mouth_vector, pose_vector
    global eyes_vector_queue, mouth_vector_queue, pose_vector_queue
    global eyes_current_score, mouth_current_score, pose_current_score
    global status, last_eyes_close, last_pose_right, last_answer_generate

    while status == 'waiting' or status == 'generating':
        eyes_score = eyes_vector_queue.pop(-1) if len(eyes_vector_queue) > 0 else eyes_current_score
        mouth_score = mouth_vector_queue.pop(-1) if len(mouth_vector_queue) > 0 else mouth_current_score
        pose_score = pose_vector_queue.pop(-1) if len(pose_vector_queue) > 0 else pose_current_score

        # random very-small eyes, mouth, pose change
        eyes_current_score = eyes_current_score + 0.07 * random.random() - 0.035
        eyes_current_score = np.clip(eyes_current_score, EYES_BASE_SCORE - 0.25, EYES_BASE_SCORE + 0.25)

        mouth_current_score = mouth_current_score + 0.05 * random.random() - 0.025
        mouth_current_score = np.clip(mouth_current_score, MOUTH_BASE_SCORE - 0.15, MOUTH_BASE_SCORE + 0.15)

        pose_current_score = pose_current_score + 0.09 * random.random() - 0.045
        pose_current_score = np.clip(pose_current_score, POSE_BASE_SCORE - 0.4, POSE_BASE_SCORE + 0.4)

        # random eyes open/close change
        if (time.time() - last_answer_generate >= 5.0 and
                (last_eyes_close is None or time.time() - last_eyes_close >= 1.0)):

            if random.random() < 0.025:
                eyes_magnitude = 1.1 + random.random() * 0.4
                r = random.random()

                if r < 0.4:
                    eyes_change_np = np.array([0.4, 0.6, 0.8, 0.8, 0.6, 0.4]) * eyes_magnitude
                elif r < 0.7:
                    eyes_change_np = np.array([0.4, 0.6, 0.8, 0.6, 0.4]) * eyes_magnitude
                else:
                    eyes_change_np = np.array([0.4, 0.7, 0.7, 0.4]) * eyes_magnitude

                eyes_change_list = list(eyes_change_np)
                eyes_vector_queue += eyes_change_list

                last_eyes_close = time.time()

        # handling long time waiting
        if status == 'waiting' and time.time() - last_answer_generate >= 120.0:
            if random.random() < 0.001:
                status = 'finished'

                ohlora_waiting_time = time.time() - last_answer_generate
                print(f'[SYSTEM MESSAGE] 오로라👱‍♀️ 의 마지막 답변 후 {int(ohlora_waiting_time)} 초 동안 '
                      f'사용자 메시지 입력이 없어서, 오로라👱‍♀️ 가 대화를 강제 종료했습니다.')
                raise Exception('finished_by_ohlora')

        # generate Oh-LoRA image
        generate_and_show_ohlora_image(stylegan_generator, ohlora_z_vector, eyes_vector, mouth_vector, pose_vector,
                                       eyes_score, mouth_score, pose_score)


# Oh-LoRA (오로라) 실행
# Create Date : 2025.05.20
# Last Update Date : 2025.05.21
# - summary 업데이트 결과가 None (7회 시도 시에도 summary LLM 답변이 형식에 맞지 않음) 일 시, summary 를 공백으로 초기화
# - S-BERT memory 항목 불러오기 위한 corr-coef threshold 0.6 -> 0.95 로 상향

# Arguments:
# - ohlora_llms           (dict(LLM))       : LLM (Polyglot-Ko 1.3B Fine-Tuned)
#                                             {'output_message': LLM, 'memory': LLM, 'summary': LLM,
#                                              'eyes_mouth_pose': LLM}
# - ohlora_llms_tokenizer (dict(tokenizer)) : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
#                                             {'output_message': tokenizer, 'memory': tokenizer, 'summary': tokenizer,
#                                              'eyes_mouth_pose': tokenizer}
# - sbert_model           (S-BERT Model)    : S-BERT (RoBERTa-based)

# Running Mechanism:
# - Oh-LoRA LLM 답변 생성 시마다 이에 기반하여 final_product/ohlora.png 경로에 오로라 이미지 생성
# - Oh-LoRA 답변을 parsing 하여 llm/memory_mechanism/saved_memory/ohlora_memory.txt 경로에 메모리 저장
# - S-BERT 모델을 이용하여, RAG 와 유사한 방식으로 해당 파일에서 사용자 프롬프트에 가장 적합한 메모리 정보를 찾아서 최종 LLM 입력에 추가

def run_ohlora(ohlora_llms, ohlora_llms_tokenizer, sbert_model):
    global ohlora_z_vector, eyes_vector, mouth_vector, pose_vector
    global status, last_answer_generate

    summary = ''
    last_answer_generate = time.time()

    thread = threading.Thread(target=realtime_ohlora_generate)
    thread.start()

    while True:
        user_prompt = input('\n오로라에게 말하기 (Ctrl+C to finish) : ')
        status = 'generating'

        # check user prompt length
        encoded_user_prompt = ohlora_llms_tokenizer['output_message'].encode(user_prompt)
        if len(encoded_user_prompt) > 48:
            print('[SYSTEM MESSAGE] 너무 긴 질문은 오로라👱‍♀️ 에게 부담 돼요! 그런 질문은 오로라의 절친 혜나 🌹 (LLM Hyena) 에게 해 주세요! 😢')
            continue

        best_memory_item = pick_best_memory_item(sbert_model,
                                                 user_prompt,
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
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments:
# - ohlora_no (int or None) : 오로라 얼굴 생성용 latent z vector 의 번호 (index, case No.)
#                             참고: 2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v7/final_OhLoRA_info.md 파일

# Returns:
# - ohlora_z_vector (NumPy array) : Oh-LoRA 이미지 생성용 latent z vector, dim = (512 + 3,)
# - eyes_vector     (NumPy array) : eyes (눈을 뜬 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - mouth_vector    (NumPy array) : mouth (입을 벌린 정도) 핵심 속성 값 변화 벡터, dim = (512,)
# - pose_vector     (NumPy array) : pose (고개 돌림) 핵심 속성 값 변화 벡터, dim = (512,)

def get_vectors(ohlora_no):
    global passed_ohlora_nos

    # find index of Oh-LoRA vectors
    ohlora_idx = None

    if ohlora_no is not None:
        for idx, passed_ohlora_no in enumerate(passed_ohlora_nos):
            if ohlora_no == passed_ohlora_no:
                ohlora_idx = idx
                break

    if ohlora_idx is None:
        ohlora_idx = random.randint(0, len(passed_ohlora_nos) - 1)
        print(f'Oh-LoRA face vector selected randomly : case No. {passed_ohlora_nos[ohlora_idx]}')

    # get Oh-LoRA vectors
    eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors()

    ohlora_z_vector_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_z_vectors.csv'
    ohlora_w_group_name_csv_path = f'{PROJECT_DIR_PATH}/stylegan/stylegan_vectorfind_v7/ohlora_w_group_names.csv'
    ohlora_z_vectors = load_ohlora_z_vectors(vector_csv_path=ohlora_z_vector_csv_path)
    ohlora_w_group_names = load_ohlora_w_group_names(group_name_csv_path=ohlora_w_group_name_csv_path)

    ohlora_z_vector = ohlora_z_vectors[ohlora_idx]
    ohlora_w_group_name = ohlora_w_group_names[ohlora_idx]

    eyes_vector = eyes_vectors[ohlora_w_group_name][0]
    mouth_vector = mouth_vectors[ohlora_w_group_name][0]
    pose_vector = pose_vectors[ohlora_w_group_name][0]

    return ohlora_z_vector, eyes_vector, mouth_vector, pose_vector


if __name__ == '__main__':

    # parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ohlora_no',
                        help="latent z vector ID for Oh-LoRA face image generation (index, case No.)",
                        default='none')
    args = parser.parse_args()

    ohlora_no = args.ohlora_no
    try:
        ohlora_no = int(ohlora_no)
    except:
        ohlora_no = None

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # get Oh-LoRA vectors
    ohlora_z_vector, eyes_vector, mouth_vector, pose_vector = get_vectors(ohlora_no)

    # load model
    stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model = load_models()
    print('ALL MODELS for Oh-LoRA (오로라) load successful!! 👱‍♀️')

    # run Oh-LoRA (오로라)
    try:
        run_ohlora(ohlora_llms, ohlora_llms_tokenizer, sbert_model)

    except KeyboardInterrupt:
        print('[SYSTEM MESSAGE] 오로라와의 대화가 끝났습니다. 👱‍♀️👋 다음에도 오로라와 함께해 주실 거죠?')
        status = 'finished'

    except Exception as e:
        print(f'error : {e}')
        status = 'finished'
