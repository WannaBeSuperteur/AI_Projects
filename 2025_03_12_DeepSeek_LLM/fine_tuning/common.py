import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ast import literal_eval
from create_dataset.common import generate_dl_model_llm_output, generate_flow_chart_llm_output
from draw_diagram.diagram_format_finder import find_diagram_formats, add_diagram_info

from collections import Counter
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
# Last Update Date : 2025.03.22
# - 평균 연결 node 개수를 비교하여 산출하는 점수를 반영

# Arguments:
# - output_data     (str) : LLM 의 출력 답변
# - llm_dest_output (str) : LLM 의 목표 답변 (정답)

# Returns:
# - score (float) : output_data 의 적절성 평가 score (0.0 - 1.0)

def compute_dl_model_task_score(output_data, llm_dest_output):
    connected_count_iou_score = compute_connected_count_iou_score(output_data, llm_dest_output)
    average_connected_nodes_score = compute_average_connected_nodes_score(output_data, llm_dest_output)

    if average_connected_nodes_score is None:
        return connected_count_iou_score

    else:
        return (connected_count_iou_score + average_connected_nodes_score) / 2.0


# Flow-Chart task 에서의 LLM 의 출력값과 정답 비교하여 점수 산출
# Create Date : 2025.03.21
# Last Update Date : 2025.03.22
# - 함수명 변경 반영

# Arguments:
# - output_data     (str) : LLM 의 출력 답변
# - llm_dest_output (str) : LLM 의 목표 답변 (정답)

# Returns:
# - score (float) : output_data 의 적절성 평가 score (0.0 - 1.0)

def compute_flowchart_task_score(output_data, llm_dest_output):
    return compute_connected_count_iou_score(output_data, llm_dest_output)


# LLM 의 출력값과 정답 비교하여 점수를 산출하기 위해, diagram_dict 에 각 node 별 해당 node와 연결된 node 개수 추가
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - diagram_dict (dict) : Diagram 관련 정보가 있는 dict

# Returns:
# - diagram_dict 에 각 node 별 해당 node와 연결된 node 개수 추가

def add_connected_node_count(diagram_dict):
    for node_id, node_info in diagram_dict.items():
        connected_node_ids = node_info['connected_nodes']
        connected_node_count = []

        for connected_node_id in connected_node_ids:
            try:
                con_node_id = int(connected_node_id)
                if con_node_id in diagram_dict.keys():
                    con_node_neighbors = diagram_dict[con_node_id]['connected_nodes']
                else:
                    con_node_neighbors = diagram_dict[str(con_node_id)]['connected_nodes']

                con_node_neighbors = list(filter(lambda x: x != '', con_node_neighbors))
                connected_node_count.append(len(con_node_neighbors))

            except:
                pass

        diagram_dict[node_id]['connected_node_count'] = connected_node_count


# LLM 의 출력값과 정답의 도형 및 그 연결 관계를 분석하기 위한 Diagram Dict 생성
# Create Date : 2025.03.22
# Last Update Date : -

# Arguments:
# - output_data     (str) : LLM 의 출력 답변
# - llm_dest_output (str) : LLM 의 목표 답변 (정답)

# Returns:
# - output_diagram_dict (dict) : LLM 의 출력 답변에 대한 diagram dict
# - dest_diagram_dict (dict) : LLM 의 목표 답변 (정답) 에 대한 diagram dict

def create_diagram_dict(output_data, llm_dest_output):
    output_diagram_dict = {}
    dest_diagram_dict = {}

    output_lines = output_data.split('\n')
    for output_line in output_lines:
        diagram_formats = find_diagram_formats(output_line)
        try:
            add_diagram_info(diagram_formats, output_diagram_dict)
        except Exception as e:
            print(f'error (output line parsing) : {e}')

    dest_lines = llm_dest_output.split('\n')
    for dest_line in dest_lines:
        diagram_formats = find_diagram_formats(dest_line)
        try:
            add_diagram_info(diagram_formats, dest_diagram_dict)
        except Exception as e:
            print(f'error (dest line parsing) : {e}')

    return output_diagram_dict, dest_diagram_dict


# LLM 의 출력값과 정답에 대해 connected node 개수의 IoU 값을 이용하여 점수 산출
# Create Date : 2025.03.21
# Last Update Date : 2025.03.22
# - 함수명 변경
# - Diagram Dict 생성 함수 분리

# Arguments:
# - output_data     (str) : LLM 의 출력 답변
# - llm_dest_output (str) : LLM 의 목표 답변 (정답)

# Returns:
# - score (float) : output_data 의 적절성 평가 score (0.0 - 1.0)

def compute_connected_count_iou_score(output_data, llm_dest_output):

    # find diagram formats and represent as dictionary
    output_diagram_dict, dest_diagram_dict = create_diagram_dict(output_data, llm_dest_output)

    # connected node count
    add_connected_node_count(output_diagram_dict)
    add_connected_node_count(dest_diagram_dict)

    # compute score using IoU Score of node count distribution
    output_connected_node_count = [str(shape_info['connected_node_count']) for shape_info in output_diagram_dict.values()]
    dest_connected_node_count = [str(shape_info['connected_node_count']) for shape_info in dest_diagram_dict.values()]

    output_connected_node_count_ = dict(Counter(output_connected_node_count))
    dest_connected_node_count_ = dict(Counter(dest_connected_node_count))
    all_connected_node_count = list(set(output_connected_node_count_.keys()) | set(dest_connected_node_count_.keys()))

    only_in_output = 0
    only_in_dest = 0
    both_in = 0

    for cnt_case in all_connected_node_count:
        cnt_case_output = output_connected_node_count_.get(cnt_case, 0)
        cnt_case_dest = dest_connected_node_count_.get(cnt_case, 0)

        if cnt_case_output > cnt_case_dest:
            only_in_output += cnt_case_output - cnt_case_dest
            both_in += cnt_case_dest
        else:
            only_in_dest += cnt_case_dest - cnt_case_output
            both_in += cnt_case_output

    iou_score = both_in / (only_in_output + only_in_dest + both_in)
    return iou_score


# LLM의 출력값과 정답에 대해 평균 연결 node 개수를 비교하여 점수 산출 (Deep Learning Model 용)
# Create Date : 2025.03.22
# Last Update Date : -

# Arguments:
# - output_data     (str) : LLM 의 출력 답변
# - llm_dest_output (str) : LLM 의 목표 답변 (정답)

# Returns:
# - score (float) : output_data 의 적절성 평가 score (0.0 - 1.0)

def compute_average_connected_nodes_score(output_data, llm_dest_output):

    # find diagram formats and represent as dictionary
    output_diagram_dict, dest_diagram_dict = create_diagram_dict(output_data, llm_dest_output)

    # connected node count
    output_connected_nodes = [len(node_info['connected_nodes']) for node_info in output_diagram_dict.values()]
    dest_connected_nodes = [len(node_info['connected_nodes']) for node_info in dest_diagram_dict.values()]

    output_cn_mean = np.mean(output_connected_nodes)
    output_cn_std = np.std(output_connected_nodes)

    dest_cn_mean = np.mean(dest_connected_nodes)
    dest_cn_std = np.std(dest_connected_nodes)

    # divide by zero 가 되는 case 의 경우 해당 점수는 무의미
    if output_cn_std == 0 or dest_cn_std == 0:
        return None

    output_z_according_to_dest = (output_cn_mean - dest_cn_mean) / dest_cn_std
    dest_z_according_to_output = (dest_cn_mean - output_cn_mean) / output_cn_std
    z_diff = abs(output_z_according_to_dest - dest_z_according_to_output)

    return max(3.0 - z_diff, 0) / 3.0


# DataFrame 에 학습 가능한 'text' column 추가
# Create Date : 2025.03.20
# Last Update Date : 2025.03.21
# - tokenizer 의 eos token 추가

# Arguments:
# - df        (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame
#                                  columns: ['input_data', 'output_data', 'dest_shape_info']

# Returns:
# - df 의 'text' column 추가
# - 해당 column 의 형식은 LLM 이 직접 학습 가능한 형태임

def add_text_column_for_llm(df):
    df['text'] = df.apply(lambda x: f"### Question: {x['input_data']}\n ### Answer: {x['output_data']}<eos>",
                          axis=1)


# for test

if __name__ == '__main__':
    output_data_0 = """
        [0, 250, 50, 'rectangle', 75, 75, 'solid arrow', (192, 224, 208), (32, 32, 32), [1]]
    [1, 500, 150, 'circle', 50, 50, 'solid arrow', (208, 208, 208), (32, 32, 64), [2]]
    [2, 500, 250, 'rectangle', 25, 25, 'solid arrow', (208, 192, 224), (64, 48, 32), [3]]
    [3, 500, 350, 'round rectangle', 50, 50, 'solid arrow', (224, 240, 232), (0, 0, 0), [4]]
    [4, 500, 450, 'rectangle', 75, 75, 'dashed line', (232, 224, 240), (32, 32, 32), [5]]
    [5, 500, 550, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (32, 32, 32), []]
    [6, 750, 250, 'round rectangle', 50, 50, 'dashed line', (255, 255, 255), (96, 96, 96), [2, 3]]
    [7, 750, 50, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (64, 128, 96), [6]]
    [8, 750, 150, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (64, 128, 96), [6]]
    [9, 750, 50, 'circle', 75, 75, 'solid arrow', (208, 192, 224), (0, 0, 0), [6]]
    [10, 750, 150, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (64, 128, 96), [6]]
    [11, 500, 350, 'rectangle', 25, 25, 'dashed line', (224, 224, 240), (64, 64, 64), [4]]
    [12, 500, 550, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (64, 128, 96), [5]]
    [13, 500, 550, 'round rectangle', 50, 50, 'dashed line', (224, 240, 232), (0, 0, 0), [5]]
    [14, 500, 250, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (64, 128, 96), [3]]
    [15, 750, 50, 'round rectangle', 50, 50, 'solid arrow', (224, 224, 224), (48, 32, 64), [3]]
    [16, 750, 250, 'round rectangle', 75, 75, 'dashed line', (224, 208, 192), (64, 48, 32), [15]]
    [17, 93.404, 50, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (64, 128, 96), [6]]
    [18, 93.404, 150, 'circle', 75, 75, 'solid arrow', (208, 192, 224), (0, 0, 0), [6]]
    [19, 750, 50, 'rectangle', 75, 75, 'solid arrow', (208, 192, 224), (48, 32, 64), [6]]
    [20, 500, 550, 'circle', 25, 25, 'solid arrow', (192, 224, 208), (64, 128, 96), [5]]
    [21, 500, 550, 'round rectangle', 50, 50, 'dashed line', (224, 224, 240), (96, 96, 96), [5]]
        """

    llm_dest_output_0 = """
        [0, 62, 300, 'round rectangle', 40, 40, 'solid arrow', (224, 224, 240), (64, 64, 128), [1]]
    [1, 187, 300, 'rectangle', 20, 20, 'solid arrow', (255, 255, 255), (0, 0, 0), [2]]
    [2, 312, 300, 'rectangle', 40, 40, 'solid arrow', (192, 192, 224), (32, 32, 64), [3, 8, 9]]
    [3, 437, 100, 'rectangle', 20, 20, 'solid arrow', (255, 255, 255), (0, 0, 0), [4]]
    [4, 562, 100, 'rectangle', 40, 40, 'solid arrow', (192, 192, 224), (32, 32, 64), [5]]
    [5, 687, 200, 'rectangle', 20, 20, 'solid arrow', (255, 255, 255), (0, 0, 0), [6, 11]]
    [6, 812, 200, 'round rectangle', 40, 40, 'dashed line', (224, 224, 240), (64, 64, 128), [7]]
    [7, 937, 300, 'circle', 68, 68, 'dashed line', (224, 224, 240), (0, 0, 0), []]
    [8, 437, 300, 'rectangle', 20, 20, 'solid arrow', (232, 224, 240), (64, 128, 96), []]
    [9, 437, 499, 'round rectangle', 40, 40, 'solid arrow', (224, 224, 240), (64, 64, 128), [10]]
    [10, 562, 300, 'circle', 20, 20, 'solid arrow', (192, 224, 208), (96, 96, 96), []]
    [11, 812, 400, 'rectangle', 68, 68, 'dashed line', (224, 224, 224), (64, 64, 64), []]
    [12, 687, 400, 'round rectangle', 40, 40, 'dashed line', (224, 224, 240), (64, 64, 128), [6]]
    [13, 562, 499, 'rectangle', 20, 20, 'solid arrow', (255, 255, 255), (0, 0, 0), [12]]
        """

    score = compute_connected_count_iou_score(output_data_0, llm_dest_output_0)
    print(score)

    output_data_1 = """
        [0, 275, 120, 'circle', 62, 75, 'solid arrow', (232, 224, 240), (32, 64, 48), [5, 6, 7, 8, 9, 10, 11]]
    [1, 385, 120, 'circle', 62, 75, 'solid arrow', (232, 224, 240), (32, 64, 48), [5, 6, 7, 8, 9, 10, 11]]
    [2, 500, 120, 'circle', 62, 75, 'solid arrow', (232, 224, 240), (32, 64, 48), [5, 6, 7, 8, 9, 10, 11]]
    [3, 615, 120, 'circle', 62, 75, 'solid arrow', (232, 224, 240), (32, 64, 48), [5, 6, 7, 8, 9, 10, 11]]
    [4, 730, 120, 'circle', 62, 75, 'solid arrow', (232, 224, 240), (32, 64, 48), [5, 6, 7, 8, 9, 10, 11]]
    [5, 275, 240, 'circle', 62, 75, 'solid arrow', (208, 208, 208), (32, 64, 48), [12, 13, 14]]
    [6, 385, 240, 'circle', 62, 75, 'solid arrow', (208, 208, 208), (32, 64, 48), [12, 13, 14]]
    [7, 500, 240, 'circle', 62, 75, 'solid arrow', (208, 208, 208), (32, 64, 48), [12, 13, 14]]
    [8, 615, 240, 'circle', 62, 75, 'solid arrow', (208, 208, 208), (32, 64, 48), [12, 13, 14]]
    [9, 730, 240, 'circle', 62, 75, 'solid arrow', (208, 208, 208), (32, 64, 48), [12, 13, 14]]
    [10, 500, 360, 'circle', 62, 75, 'solid arrow', (208, 208, 208), (32, 64, 48), [12, 13, 14]]
    [11, 615, 360, 'circle', 62, 75, 'solid arrow', (208, 208, 208), (32, 64, 48), [12, 13, 14]]
    [12, 325, 480, 'circle', 105, 75, 'solid arrow', (224, 224, 240), (64, 64, 64), []]
    [13, 500, 480, 'circle', 105, 75, 'solid arrow', (224, 224, 240), (64, 64, 64), []]
    [14, 675, 480, 'circle', 105, 75, 'solid arrow', (224, 224, 240), (64, 64, 64), []]
    [15, 500, 599, 'circle', 105, 75, 'solid arrow', (208, 208, 208), (32, 32, 32), []]
    [16, 422, 599, 'circle', 105, 75, 'solid arrow', (208, 208, 208), (32, 32, 32), [12, 13, 14]]
    [17, 333, 599, 'circle', 105, 75, 'solid arrow', (208, 208, 208), (32, 32, 32), [12, 13, 14]]
    [18, 244, 599, 'circle', 105, 75, 'solid arrow', (208, 208, 208), (32, 32, 32), [12, 13, 14]]
        """

    llm_dest_output_1 = """
        [0, 250, 150, 'circle', 75, 100, 'solid arrow', (208, 192, 224), (96, 64, 128), [5, 6, 7, 8]]
    [1, 375, 150, 'circle', 75, 100, 'solid arrow', (208, 192, 224), (96, 64, 128), [5, 6, 7, 8]]
    [2, 500, 150, 'circle', 75, 100, 'solid arrow', (208, 192, 224), (96, 64, 128), [5, 6, 7, 8]]
    [3, 625, 150, 'circle', 75, 100, 'solid arrow', (208, 192, 224), (96, 64, 128), [5, 6, 7, 8]]
    [4, 750, 150, 'circle', 75, 100, 'solid arrow', (208, 192, 224), (96, 64, 128), [5, 6, 7, 8]]
    [5, 298, 300, 'circle', 84, 100, 'solid arrow', (224, 240, 232), (64, 64, 128), [9, 10]]
    [6, 432, 300, 'circle', 84, 100, 'solid arrow', (224, 240, 232), (64, 64, 128), [9, 10]]
    [7, 567, 300, 'circle', 84, 100, 'solid arrow', (224, 240, 232), (64, 64, 128), [9, 10]]
    [8, 701, 300, 'circle', 84, 100, 'solid arrow', (224, 240, 232), (64, 64, 128), [9, 10]]
    [9, 423, 450, 'circle', 114, 100, 'solid arrow', (224, 240, 232), (64, 48, 32), []]
    [10, 576, 450, 'circle', 114, 100, 'solid arrow', (224, 240, 232), (64, 48, 32), []]
        """

    score = compute_average_connected_nodes_score(output_data_1, llm_dest_output_1)
    print(score)
