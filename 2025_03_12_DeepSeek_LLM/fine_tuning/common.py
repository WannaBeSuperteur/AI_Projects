import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ast import literal_eval
from create_dataset.common import generate_dl_model_llm_output, generate_flow_chart_llm_output

import pandas as pd
import numpy as np


# 각 LLM output 의 score 평가 (LLM 평가 또는 ORPO 용 데이터셋 생성할 때 사용 가능)
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - shape_info  (dict) : LLM 의 출력 답변을 통해 생성해야 할 도형의 정보
# - output_data (str)  : LLM 의 출력 답변

# Returns:
# - score (float) : input_data 에 대한 output_data 의 적절성 평가 score (0.0 - 1.0)

def compute_output_score(shape_info, output_data):
    shape_info_dict = literal_eval(shape_info)

    task_name = shape_info_dict['task_name']
    shape_types = shape_info_dict['shape_types']
    shape_sizes = shape_info_dict['shape_sizes']

    if task_name == 'dl_model':
        llm_dest_output = generate_dl_model_llm_output(shape_types, shape_sizes)
        return compute_dl_model_task_score(output_data, llm_dest_output)

    else:  # flowchart
        llm_dest_output = generate_flow_chart_llm_output(shape_types, shape_sizes)
        return compute_flowchart_task_score(output_data, llm_dest_output)


# Deep Learning Model task 에서의 LLM 의 출력값과 정답 비교하여 점수 산출
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - output_data     (str) : LLM 의 출력 답변
# - llm_dest_output (str) : LLM 의 목표 답변 (정답)

# Returns:
# - score (float) : output_data 의 적절성 평가 score (0.0 - 1.0)

def compute_dl_model_task_score(output_data, llm_desst_output):
    return 0.5  # temp


# Flow-Chart task 에서의 LLM 의 출력값과 정답 비교하여 점수 산출
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - output_data     (str) : LLM 의 출력 답변
# - llm_dest_output (str) : LLM 의 목표 답변 (정답)

# Returns:
# - score (float) : output_data 의 적절성 평가 score (0.0 - 1.0)

def compute_flowchart_task_score(output_data, llm_desst_output):
    return 0.75  # temp


# DataFrame 에 학습 가능한 'text' column 추가
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - df        (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame
#                                  columns: ['input_data', 'output_data', 'dest_shape_info']

# Returns:
# - df 의 'text' column 추가
# - 해당 column 의 형식은 LLM 이 직접 학습 가능한 형태임

def add_text_column_for_llm(df):
    df['text'] = df.apply(lambda x: f"### Question: {x['input_data']}\n ### Answer: {x['output_data']}",
                          axis=1)
