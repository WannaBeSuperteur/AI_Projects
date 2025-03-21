from sft_fine_tuning import load_sft_llm
from common import compute_output_score
import pandas as pd
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from draw_diagram.draw_diagram import generate_diagram_from_lines


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# ORPO 용 데이터셋 (SFT 와 동일한 포맷) 생성에 필요한 데이터 생성 (LLM 출력에 대한 평가 결과)
# Create Date : 2025.03.22
# Last Update Date : -

# Arguments:
# - llm            (LLM)              : SFT 로 Fine-tuning 된 LLM
# - tokenizer      (tokenizer)        : 해당 LLM 에 대한 tokenizer
# - orpo_format_df (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame (from create_dataset/orpo_dataset_llm.csv)
#                                       columns: ['input_data', 'output_data', 'dest_shape_info', 'score']

# Returns:
# - orpo_info_df (Pandas DataFrame) : ORPO 용 데이터셋 생성에 필요한 LLM 출력 평가 결과의 DataFrame
#                                     columns: ['input_data', 'output_data', 'score']
#
# - log/orpo_info_df.csv 에 ORPO 용 LLM 출력 평가 결과를 중간 저장

def prepare_orpo_info(llm, tokenizer, orpo_format_df):
    orpo_info_df = {'input_data': [], 'output_data': [], 'score': []}

    for idx, row in orpo_format_df.iterrows():
        prompt = row['input_data']
        dest_shape_info = row['dest_shape_info']

        inputs = tokenizer(f'### Question: {prompt}\n ### Answer: ', return_tensors='pt').to(llm.device)

        with torch.no_grad():
            outputs = llm.generate(**inputs, max_length=1536, do_sample=True)
            llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace('<|EOT|>', '')
            llm_answer = llm_answer.split('### Answer: ')[1]  # prompt 부분을 제외한 answer 만 표시
            score = compute_output_score(dest_shape_info, llm_answer)

        # llm answer 를 이용하여 Diagram 생성 및 저장
        try:
            llm_answer_lines = llm_answer.split('\n')
            os.makedirs(f'{PROJECT_DIR_PATH}/fine_tuning/sft_model_diagrams_on_orpo_dataset', exist_ok=True)
            diagram_save_path = f'{PROJECT_DIR_PATH}/fine_tuning/sft_model_diagrams_on_orpo_dataset/diagram_{idx:06d}.png'
            generate_diagram_from_lines(llm_answer_lines, diagram_save_path)

        except Exception as e:
            print(f'SFT diagram generation failed: {e}')

        orpo_info_df['input_data'].append(prompt)
        orpo_info_df['output_data'].append(llm_answer)
        orpo_info_df['score'].append(score)

        pd.DataFrame(orpo_info_df).to_csv(f'{PROJECT_DIR_PATH}/fine_tuning/log/orpo_info_df.csv')

    return orpo_info_df


# ORPO 용 데이터셋 (SFT 와 동일한 포맷) 의 DataFrame 에 ORPO 용 데이터 추가
# Create Date : 2025.03.22
# Last Update Date : -

# Arguments:
# - orpo_format_df (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame (from create_dataset/orpo_dataset_llm.csv)
#                                       columns: ['input_data', 'output_data', 'dest_shape_info', 'score']
# - orpo_info_df   (Pandas DataFrame) : ORPO 용 데이터셋 생성에 필요한 LLM 출력 평가 결과의 DataFrame
#                                       columns: ['input_data', 'output_data', 'score']

# Returns:
# - orpo_df (Pandas DataFrame) : ORPO 용 데이터가 추가된 데이터셋의 DataFrame
#                                columns: ['input_data', 'output_data', 'dest_shape_info', 'score']

def add_orpo_dataset(orpo_format_df, orpo_info_df):
    raise NotImplementedError


if __name__ == '__main__':
    llm, tokenizer = load_sft_llm()
    orpo_format_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/orpo_dataset_llm.csv'
    orpo_format_df = pd.read_csv(orpo_format_dataset_path)

    # 이미 ORPO 용 데이터가 추가된 경우 오류 반환
    assert min(orpo_format_df['score']) == 1.0, 'ORPO DATA ALREADY ADDED'

    orpo_info_df = prepare_orpo_info(llm, tokenizer, orpo_format_df)
    orpo_df = add_orpo_dataset(orpo_format_df, orpo_info_df)

    orpo_df.to_csv(f'{PROJECT_DIR_PATH}/create_dataset/orpo_dataset_llm.csv')
