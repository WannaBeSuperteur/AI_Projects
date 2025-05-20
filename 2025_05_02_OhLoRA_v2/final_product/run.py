import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from run_llm import (generate_llm_answer, clean_llm_answer, parse_memory, save_memory_list, summarize_llm_answer,
                     decide_property_scores)
from run_display import generate_ohlora_image

from stylegan.stylegan_common.stylegan_generator import StyleGANGeneratorForV6
from llm.memory_mechanism.load_sbert_model import load_pretrained_sbert_model
from llm.run_memory_mechanism import pick_best_memory_item


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


# Oh-LoRA (오로라) 실행
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments:
# - stylegan_generator    (nn.Module)       : StyleGAN-VectorFind-v7 generator
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

def run_ohlora(stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model):
    summary = ''

    while True:
        user_prompt = input('\n오로라에게 말하기 (Ctrl+C to finish) : ')
        encoded_user_prompt = ohlora_llms_tokenizer['output_message'].encode(user_prompt)

        if len(encoded_user_prompt) > 48:
            print('[SYSTEM MESSAGE] 너무 긴 질문은 오로라👱‍♀️ 에게 부담 돼요! 😢')
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

        print('best_memory_item :', best_memory_item)
        print('final_ohlora_input :', final_ohlora_input)

        # generate Oh-LoRA answer and post-process
        llm_answer = generate_llm_answer(ohlora_llm=ohlora_llms['output_message'],
                                         ohlora_llm_tokenizer=ohlora_llms_tokenizer['output_message'],
                                         final_ohlora_input=final_ohlora_input)
        llm_answer_cleaned = clean_llm_answer(llm_answer)
        print(f'👱‍♀️ 오로라 : {llm_answer_cleaned}')

        # update memory
        memory_list = parse_memory(memory_llm=ohlora_llms['memory'],
                                   memory_llm_tokenizer=ohlora_llms_tokenizer['memory'],
                                   final_ohlora_input=final_ohlora_input)
        save_memory_list(memory_list)

        # update summary
        summary = summarize_llm_answer(summary_llm=ohlora_llms['summary'],
                                       summary_llm_tokenizer=ohlora_llms_tokenizer['summary'],
                                       final_ohlora_input=final_ohlora_input,
                                       llm_answer_cleaned=llm_answer_cleaned)

        # generate Oh-LoRA image
        eyes_score, mouth_score, pose_score = decide_property_scores(
            eyes_mouth_pose_llm=ohlora_llms['eyes_mouth_pose'],
            eyes_mouth_pose_llm_tokenizer=ohlora_llms_tokenizer['eyes_mouth_pose_llm_tokenizer'],
            llm_answer_cleaned=llm_answer_cleaned)

        generate_ohlora_image(stylegan_generator, eyes_score, mouth_score, pose_score)


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # load model
    stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model = load_models()
    print('ALL MODELS for Oh-LoRA (오로라) load successful!! 👱‍♀️')

    # run Oh-LoRA (오로라)
    run_ohlora(stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model)
