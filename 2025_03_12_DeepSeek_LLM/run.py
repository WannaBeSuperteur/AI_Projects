import argparse
import os
import shutil

import torch

from common_values import PROMPT_PREFIX, PROMPT_SUFFIX
from fine_tuning.sft_fine_tuning import load_sft_llm
from draw_diagram.draw_diagram import generate_diagram_from_lines

from final_recommend_score.knn import load_test_diagrams
from final_recommend_score.final_recommend_score import (load_models,
                                                         compute_cnn_score,
                                                         compute_ae_encoder_score,
                                                         compute_final_recommend_score)

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))


# SFT 로 Fine-Tuning 된 LLM 을 실행
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - llm               (LLM)       : SFT 로 Fine-tuning 된 LLM
# - tokenizer         (tokenizer) : 해당 LLM 의 tokenizer
# - prompt            (str)       : 해당 LLM 으로 입력되는 User Prompt
# - max_answer_tokens (int)       : 해당 LLM 이 생성하는 답변의 최대 토큰 개수
# - llm_answer_count  (int)       : 생성할 LLM 답변의 개수

# Returns:
# - llm_answers (list(str)) : 해당 LLM 의 답변 리스트

def run_llm(llm, tokenizer, prompt, max_answer_tokens, llm_answer_count):
    inputs = tokenizer(f'### Question: {prompt}\n ### Answer: ', return_tensors='pt').to(llm.device)
    input_length = inputs['input_ids'].shape[1]

    llm_answers = []

    with torch.no_grad():
        for i in range(llm_answer_count):
            print(f'llm output generating ({i + 1} / {llm_answer_count}) ...')

            outputs = llm.generate(**inputs, max_length=input_length + max_answer_tokens, do_sample=True)
            llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace('<|EOT|>', '')
            llm_answer = llm_answer.split('### Answer: ')[1]  # prompt 부분을 제외한 나머지 부분

            llm_answers.append(llm_answer)

    return llm_answers


# llm answer 를 이용하여 Diagram 생성 및 저장
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - llm_answers (list(str)) : 해당 LLM 의 답변 리스트

# Returns:
# - 모델의 답변을 이용하여, user_diagrams/test_diagram_{k}.png 다이어그램 파일 생성
# - 추가적으로, user_diagrams/llm_answer_{k}.txt 에 LLM 의 답변을 각각 저장

def create_diagrams(llm_answers):

    for idx, llm_answer in enumerate(llm_answers):
        try:
            llm_answer_lines = llm_answer.split('\n')

            diagram_dir = f'{PROJECT_DIR_PATH}/user_diagrams/generated'
            os.makedirs(diagram_dir, exist_ok=True)
            diagram_save_path = f'{diagram_dir}/test_diagram_{idx:06d}.png'

            generate_diagram_from_lines(llm_answer_lines, diagram_save_path)

            # save llm answer
            llm_answer_save_path = f'{diagram_dir}/llm_answer_{idx:06d}.txt'

            f = open(llm_answer_save_path, 'w')
            f.write(llm_answer)
            f.close()

        except Exception as e:
            print(f'SFT diagram generation failed: {e}')


# User Prompt 읽기
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - prompt (str) : LLM 으로 입력되는 User Prompt (from user_prompt.txt)

def read_prompt():
    user_prompt_path = 'user_prompt.txt'

    f = open(user_prompt_path, 'r')
    lines = f.readlines()
    f.close()

    prompt = ''.join(lines)
    return prompt


# 최종 점수 (= 기본 가독성 점수 + 예상 사용자 평가 점수) 상위 R 개의 다이어그램을 추천 및 user_diagrams/recommended 로 복사
# Create Date : 2025.03.26
# Last Update Date : -

# Arguments:
# - recommend_count (int) : 추천 및 user_diagrams/recommended 로 이동시킬 Diagram 의 개수 (= R)

# Returns:
# - user_diagrams/generated 의 다이어그램 중 최종 점수 상위 R 개를 user_diagrams/recommended 로 복사 및 콘솔에 print

def recommend_diagrams(recommend_count):
    generated_diagram_path = f'{PROJECT_DIR_PATH}/user_diagrams/generated'
    recommended_diagram_path = f'{PROJECT_DIR_PATH}/user_diagrams/recommended'

    cnn_models, ae_encoder = load_models()
    test_diagrams, test_diagram_paths = load_test_diagrams(test_diagram_dir=generated_diagram_path)

    cnn_score_df = compute_cnn_score(cnn_models, test_diagrams, test_diagram_paths)
    ae_score_df = compute_ae_encoder_score(ae_encoder, test_diagrams, test_diagram_paths)

    final_recommend_score_df = compute_final_recommend_score(cnn_score_df, ae_score_df, save_df=False)

    print('\nFINAL RECOMMEND SCORE :')
    print(final_recommend_score_df)

    # recommend and copy top R diagrams
    final_recommend_score_df.sort_values(by='final_score', inplace=True, ascending=False)
    final_recommend_score_top_df = final_recommend_score_df[:recommend_count]

    print(f'\nFINAL RECOMMEND SCORE (TOP {recommend_count}) :')
    print(final_recommend_score_top_df)

    print('\nI recommend generated diagrams below, and I will copy them to user_diagrams/recommended ! 😊\n')
    os.makedirs(recommended_diagram_path, exist_ok=True)

    for idx, recommended_diagram_info in final_recommend_score_top_df.iterrows():
        diagram_img_path = recommended_diagram_info['img_path']
        diagram_final_score = recommended_diagram_info['final_score']

        print(f'{idx}. {diagram_img_path} (final score : {round(diagram_final_score, 4)} / 10)')

        diagram_img_name = diagram_img_path.split('/')[-1]
        generated_img_path = f'{generated_diagram_path}/{diagram_img_name}'
        dest_path = f'{recommended_diagram_path}/{diagram_img_name}'

        shutil.copy(generated_img_path, dest_path)


if __name__ == '__main__':

    # parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-length', help='answer length (maximum answer tokens)', default=1280)
    parser.add_argument('-count', help='llm answer count', default=5)
    parser.add_argument('-recommend', help='diagram count to recommend', default=3)

    args = parser.parse_args()

    max_answer_tokens = int(args.length)
    llm_answer_count = int(args.count)
    recommend_count = int(args.recommend)

    assert recommend_count <= llm_answer_count, 'MUST BE: recommended diagram count <= llm answer count = diagram count'

    print(f'option : max answer tokens = {max_answer_tokens}, ' +
          f'llm answers = {llm_answer_count}, ' +
          f'recommend diagrams = {recommend_count}')

    # load llm
    llm, tokenizer = load_sft_llm()
    assert llm is not None and tokenizer is not None, "Prepare LLM First."
    print('loading LLM successful!')

    # read prompt
    prompt = PROMPT_PREFIX + read_prompt() + PROMPT_SUFFIX
    print(f'User Prompt:\n{prompt}')

    # generate diagrams
    llm_answers = run_llm(llm, tokenizer, prompt, max_answer_tokens, llm_answer_count)
    create_diagrams(llm_answers)

    # recommend
    recommend_diagrams(recommend_count)
