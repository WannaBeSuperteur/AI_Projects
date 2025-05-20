import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import random
import math

import argparse
import threading
import os
import sys
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
status = 'running'

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

def handle_ohlora_answered(eyes_score, mouth_score, pose_score):
    global eyes_vector_queue, mouth_vector_queue, pose_vector_queue

    for cosine_line_value in cosine_line_values:
        eyes_vector_queue.append(EYES_BASE_SCORE + (eyes_score - EYES_BASE_SCORE) * cosine_line_value)
        mouth_vector_queue.append(MOUTH_BASE_SCORE + (mouth_score - MOUTH_BASE_SCORE) * cosine_line_value)
        pose_vector_queue.append(POSE_BASE_SCORE + (pose_score - POSE_BASE_SCORE) * cosine_line_value)


# Oh-LoRA (오로라) 실시간 이미지 생성 핸들링
# Create Date : 2025.05.20
# Last Update Date : -

def realtime_ohlora_generate():
    global ohlora_z_vector, eyes_vector, mouth_vector, pose_vector
    global eyes_vector_queue, mouth_vector_queue, pose_vector_queue
    global status

    while status == 'running':
        eyes_score = eyes_vector_queue.pop(-1) if len(eyes_vector_queue) > 0 else EYES_BASE_SCORE
        mouth_score = mouth_vector_queue.pop(-1) if len(mouth_vector_queue) > 0 else MOUTH_BASE_SCORE
        pose_score = pose_vector_queue.pop(-1) if len(pose_vector_queue) > 0 else POSE_BASE_SCORE

        generate_and_show_ohlora_image(stylegan_generator, ohlora_z_vector, eyes_vector, mouth_vector, pose_vector,
                                       eyes_score, mouth_score, pose_score)


# Oh-LoRA (오로라) 실행
# Create Date : 2025.05.20
# Last Update Date : -

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

    summary = ''

    thread = threading.Thread(target=realtime_ohlora_generate)
    thread.start()

    while True:
        user_prompt = input('\n오로라에게 말하기 (Ctrl+C to finish) : ')

        # check user prompt length
        encoded_user_prompt = ohlora_llms_tokenizer['output_message'].encode(user_prompt)
        if len(encoded_user_prompt) > 48:
            print('[SYSTEM MESSAGE] 너무 긴 질문은 오로라👱‍♀️ 에게 부담 돼요! 그런 질문은 오로라의 절친 혜나 🌹 (LLM Hyena) 에게 해 주세요! 😢')
            continue

        best_memory_item = pick_best_memory_item(sbert_model,
                                                 user_prompt,
                                                 memory_file_name='ohlora_memory.txt',
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

        # generate Oh-LoRA image
        eyes_score_text, mouth_score_text, pose_score_text = decide_property_score_texts(
            eyes_mouth_pose_llm=ohlora_llms['eyes_mouth_pose'],
            eyes_mouth_pose_llm_tokenizer=ohlora_llms_tokenizer['eyes_mouth_pose'],
            llm_answer_cleaned=llm_answer_cleaned)

        eyes_score, mouth_score, pose_score = decide_property_scores(eyes_score_text, mouth_score_text, pose_score_text)

        print(f'👱‍♀️ 오로라 : {llm_answer_cleaned}')
        handle_ohlora_answered(eyes_score, mouth_score, pose_score)


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
        print('Oh-LoRA Finished. Good bye! 👱‍♀️👋')
        status = 'finished'
