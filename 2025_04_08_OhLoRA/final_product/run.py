import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList

import pandas as pd
import numpy as np

import time
import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from stylegan_and_segmentation.stylegan_modified.stylegan_generator import StyleGANGeneratorForV3
from llm.memory_mechanism.train_sbert import load_pretrained_sbert_model
from llm.run_memory_mechanism import pick_best_memory_item
from llm.fine_tuning.inference import StopOnTokens, load_valid_user_prompts


global_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(global_path)

from global_common.visualize_tensor import save_tensor_png


PROPERTY_DIMS_Z = 7


# 필요한 모델 로딩 : StyleGAN-FineTune-v3 Generator (Decoder), LLM (Polyglot-Ko 1.3B Fine-Tuned), S-BERT (RoBERTa-based)
# Create Date : 2025.04.23
# Last Update Date : -

# Arguments:
# - device (device) : 모델들을 mapping 시킬 device (GPU 등)

# Returns:
# - stylegan_generator   (nn.Module)    : StyleGAN-FineTune-v3 Generator (Decoder)
# - ohlora_llm           (LLM)          : LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - ohlora_llm_tokenizer (tokenizer)    : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
# - sbert_model          (S-BERT Model) : S-BERT (RoBERTa-based)

def load_models(device):

    # load StyleGAN-FineTune-v3 generator model
    stylegan_generator = StyleGANGeneratorForV3(resolution=256)
    stylegan_modified_dir = f'{PROJECT_DIR_PATH}/stylegan_and_segmentation/stylegan_modified'
    generator_path = f'{stylegan_modified_dir}/stylegan_gen_fine_tuned_v3_ckpt_0005_gen.pth'

    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    stylegan_generator.load_state_dict(generator_state_dict)
    stylegan_generator.to(device)

    # load Oh-LoRA final LLM and tokenizer
    ohlora_llm = AutoModelForCausalLM.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/polyglot_fine_tuned',
                                                      trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16).cuda()

    ohlora_llm_tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_DIR_PATH}/llm/models/polyglot_fine_tuned')

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    ohlora_llm.generation_config.pad_token_id = ohlora_llm_tokenizer.pad_token_id

    # load S-BERT Model (RoBERTa-based)
    model_path = f'{PROJECT_DIR_PATH}/llm/models/memory_sbert/trained_sbert_model'
    sbert_model = load_pretrained_sbert_model(model_path)

    return stylegan_generator, ohlora_llm, ohlora_llm_tokenizer, sbert_model


# Oh-LoRA (오로라) 실행
# Create Date : 2025.04.23
# Last Update Date : -

# Arguments:
# - stylegan_generator   (nn.Module)    : StyleGAN-FineTune-v3 Generator (Decoder)
# - ohlora_llm           (LLM)          : LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - ohlora_llm_tokenizer (tokenizer)    : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
# - sbert_model          (S-BERT Model) : S-BERT (RoBERTa-based)

# Running Mechanism:
# - Oh-LoRA LLM 답변 생성 시마다 이에 기반하여 final_product/ohlora.png 경로에 오로라 이미지 생성
# - Oh-LoRA 답변을 parsing 하여 llm/memory_mechanism/saved_memory/ohlora_memory.txt 경로에 메모리 저장
# - S-BERT 모델을 이용하여, RAG 와 유사한 방식으로 해당 파일에서 사용자 프롬프트에 가장 적합한 메모리 정보를 찾아서 최종 LLM 입력에 추가

def run_ohlora(stylegan_generator, ohlora_llm, ohlora_llm_tokenizer, sbert_model):

    while True:
        user_prompt = input('\n오로라에게 말하기 (Ctrl+C to finish) : ')
        best_memory_item = pick_best_memory_item(sbert_model,
                                                 user_prompt,
                                                 memory_file_name='ohlora_memory.txt',
                                                 verbose=False)

        if best_memory_item == '':
            final_ohlora_input = user_prompt
        else:
            final_ohlora_input = best_memory_item + ' ' + user_prompt

        # generate Oh-LoRA answer and post-process
        llm_answer = generate_llm_answer(ohlora_llm, ohlora_llm_tokenizer, final_ohlora_input)
        llm_answer_cleaned, memory_list = parse_memory(llm_answer)
        save_memory_list(memory_list)

        print(f'👱‍♀️ 오로라 : {llm_answer_cleaned}')

        # generate Oh-LoRA image
        eyes_score, mouth_score, pose_score = decide_property_scores(llm_answer_cleaned)
        generate_ohlora_image(stylegan_generator, eyes_score, mouth_score, pose_score)


# Oh-LoRA (오로라) 의 답변 생성
# Create Date : 2025.04.23
# Last Update Date : 2025.04.24
# - 폭 없는 공백 (zwsp), 줄 바꿈 없는 공백 (nbsp) 제거 처리 추가

# Arguments :
# - ohlora_llm           (LLM)       : LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - ohlora_llm_tokenizer (tokenizer) : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
# - final_ohlora_input   (str)       : 오로라👱‍♀️ 에게 최종적으로 입력되는 메시지 (경우에 따라 memory text 포함)

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
                                      max_length=80,
                                      do_sample=True,
                                      temperature=0.6,
                                      stopping_criteria=stopping_criteria)

        llm_answer = ohlora_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_answer = llm_answer[len(final_ohlora_input_):]
        llm_answer = llm_answer.replace('\u200b','').replace('\xa0', '')  # zwsp, nbsp (폭 없는 공백, 줄 바꿈 없는 공백) 제거
        trial_count += 1

        # check LLM answer and return or retry
        is_bracketed = llm_answer.startswith('[') and llm_answer.endswith(']')
        is_non_empty = (not is_bracketed) and llm_answer.replace('\n', '').replace('(답변 종료)', '').replace(' ', '') != ''
        is_answer_end_mark = '답변 종료' in llm_answer.replace('(답변 종료)', '') or '답변종료' in llm_answer.replace('(답변 종료)', '')

        is_unnecessary_quote = '"' in llm_answer or '”' in llm_answer or '“' in llm_answer or '’' in llm_answer
        is_unnecessary_mark = '�' in llm_answer
        is_too_many_blanks = '     ' in llm_answer
        is_low_quality = is_unnecessary_quote or is_unnecessary_mark or is_too_many_blanks

        if is_non_empty and not is_answer_end_mark and 'http' not in llm_answer and not is_low_quality:
            if llm_answer.startswith(']'):
                return llm_answer[1:].replace('(답변 종료)', '')
            return llm_answer.replace('(답변 종료)', '')

    return '(읽씹)'


# Oh-LoRA (오로라) 의 생성된 답변으로부터 memory 정보를 parsing
# Create Date : 2025.04.23
# Last Update Date : -

# Arguments :
# - llm_answer (str) : 오로라👱‍♀️ 가 생성한 원본 답변 (memory info 포함 가능)

# Returns :
# - llm_answer_cleaned (str)       : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변
# - memory_list        (list(str)) : 오로라👱‍♀️ 가 저장해야 할 메모리 목록

def parse_memory(llm_answer):
    llm_answer_length = len(llm_answer)

    bracket_start_or_sentence_end_idx = llm_answer_length
    bracket_end_idx = None
    is_colon = False

    llm_answer_cleaned = ''
    memory_list = []

    for i in range(llm_answer_length-1, -1, -1):
        if llm_answer[i] == ']':
            bracket_end_idx = i

        if bracket_end_idx is not None and llm_answer[i] == ':':
            is_colon = True

        if llm_answer[i] == '[' and bracket_end_idx is not None:
            llm_answer_cleaned = llm_answer[bracket_end_idx+1:bracket_start_or_sentence_end_idx] + llm_answer_cleaned

            if is_colon:
                memory_list.append(llm_answer[i:bracket_end_idx+1])

            bracket_end_idx = None
            bracket_start_or_sentence_end_idx = i
            is_colon = False

    llm_answer_cleaned = llm_answer[:bracket_start_or_sentence_end_idx] + llm_answer_cleaned
    return llm_answer_cleaned, memory_list


# Oh-LoRA (오로라) 의 메모리 정보를 llm/memory_mechanism/saved_memory/ohlora_memory.txt 에 저장
# Create Date : 2025.04.23
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


# Oh-LoRA (오로라) 의 답변에 따라 눈을 뜬 정도 (eyes), 입을 벌린 정도 (mouth), 고개 돌림 (pose) 점수 산출
# Create Date : 2025.04.23
# Last Update Date : 2025.04.24
# - 눈을 크게 뜨고 입을 벌리는 감탄사 조건 일부 수정
# - 고개 돌림 조건 로직 일부 수정

# Arguments :
# - llm_answer_cleaned (str) : 오로라👱‍♀️ 가 생성한 원본 답변에서 text clean 을 실시한 이후의 답변

# Returns :
# - eyes_score  (float) : 눈을 뜬 정도 (eyes) 의 속성 값 점수
# - mouth_score (float) : 입을 벌린 정도 (mouth) 의 속성 값 점수
# - pose_score  (float) : 고개 돌림 (pose) 의 속성 값 점수

def decide_property_scores(llm_answer_cleaned):

    if (('오!' in llm_answer_cleaned) or ('와!' in llm_answer_cleaned) or
        ('와우' in llm_answer_cleaned) or (llm_answer_cleaned.strip().startswith('오 '))):

        eyes_score = 1.6
        mouth_score_bonus = 1.2

    elif ('아!' in llm_answer_cleaned) or (llm_answer_cleaned.strip().startswith('아 ')):
        eyes_score = 0.8
        mouth_score_bonus = 0.6

    else:
        eyes_score = -1.2
        mouth_score_bonus = 0.0

    if '!' in llm_answer_cleaned or '?' in llm_answer_cleaned:
        mouth_score = 0.3 + 0.3 * min(llm_answer_cleaned.count('!'), 3) + mouth_score_bonus / 3
    else:
        mouth_score = -1.2 + mouth_score_bonus

    if '미안' in llm_answer_cleaned and '싫어' in llm_answer_cleaned:
        pose_score = 3.6
    elif '오!' not in llm_answer_cleaned and (' 못 ' in llm_answer_cleaned or '없어 ' in llm_answer_cleaned or '없어.' in llm_answer_cleaned):
        pose_score = 3.0
    elif '미안' in llm_answer_cleaned or '싫어' in llm_answer_cleaned or '별로' in llm_answer_cleaned:
        pose_score = 2.4
    elif '…' in llm_answer_cleaned:
        pose_score = 0.6
    else:
        pose_score = -1.2

    return eyes_score, mouth_score, pose_score


# Oh-LoRA (오로라) 이미지 생성
# Create Date : 2025.04.23
# Last Update Date : 2025.04.24
# - noise 강도 수정 (0.3 -> 0.15)

# Arguments :
# - stylegan_generator (nn.Module) : StyleGAN-FineTune-v3 Generator (Decoder)
# - eyes_score         (float)     : 눈을 뜬 정도 (eyes) 의 속성 값 점수
# - mouth_score        (float)     : 입을 벌린 정도 (mouth) 의 속성 값 점수
# - pose_score         (float)     : 고개 돌림 (pose) 의 속성 값 점수

# Returns :
# - 직접 반환되는 값 없음
# - final_product/ohlora.png 경로에 오로라 이미지 생성

def generate_ohlora_image(stylegan_generator, eyes_score, mouth_score, pose_score):

    # z vector 로딩
    z_vector_csv_path = f'{PROJECT_DIR_PATH}/final_product/final_ohlora_z_vector.csv'
    z_vector = pd.read_csv(z_vector_csv_path)
    z_vector_torch = torch.tensor(np.array(z_vector))

    # StyleGAN-FineTune-v3 에 입력할 속성 값을 나타내는 label 생성
    label = [eyes_score, 0.0, 0.0, mouth_score, pose_score, 0.0, 0.0]

    label_np = np.array([[label]])
    label_np = label_np.reshape((1, PROPERTY_DIMS_Z))
    label_torch = torch.tensor(label_np).to(torch.float32)

    with torch.no_grad():
        z = z_vector_torch.to(torch.float32)
        z_noised = z + 0.15 * torch.randn_like(z)

        generated_image = stylegan_generator(z=z_noised.cuda(), label=label_torch.cuda())['image']
        generated_image = generated_image.detach().cpu()

    save_tensor_png(generated_image[0],
                    image_save_path=f'{PROJECT_DIR_PATH}/final_product/ohlora.png')


# 답변 생성 테스트 (테스트용 함수, 실제 사용 시에는 해당 함수 실행 부분 주석 처리)
# Create Date : 2025.04.24
# Last Update Date : -

# Arguments:
# - ohlora_llm           (LLM)       : LLM (Polyglot-Ko 1.3B Fine-Tuned)
# - ohlora_llm_tokenizer (tokenizer) : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer

def test_generate(ohlora_llm, ohlora_llm_tokenizer):
    valid_user_prompts = load_valid_user_prompts()

    for user_prompt in valid_user_prompts:
        print(f'\nuser prompt : [{user_prompt}]')

        for _ in range(4):
            start_at = time.time()
            llm_answer = generate_llm_answer(ohlora_llm, ohlora_llm_tokenizer, user_prompt)
            elapsed_time = time.time() - start_at

            print(f'Oh-LoRA 👱‍♀️ : [{llm_answer}] (🕚 {elapsed_time:.2f}s)')


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # load model
    stylegan_generator, ohlora_llm, ohlora_llm_tokenizer, sbert_model = load_models(device)
    print('ALL MODELS for Oh-LoRA (오로라) load successful!! 👱‍♀️')

    # test generating answer
#    test_generate(ohlora_llm, ohlora_llm_tokenizer)

    # run Oh-LoRA (오로라)
    run_ohlora(stylegan_generator, ohlora_llm, ohlora_llm_tokenizer, sbert_model)
