
######## SUMMARY (BIRD-EYE VIEW) ########

"""
This common.py contains functions for:

Generating Training Dataset, that is, User Prompts (LLM input) and LLM outputs,
for LLM SFT (Supervised Fine-tuning) and ORPO (Odd-Ratio Preference Optimization),
for the two diagram types below:

 - 1. Deep Learning Model (Dense, CNN)
 - 2. Flow-Chart

 (int)                      (process)
 [ shape config seed ] ---> ( generate_structure_func )
                                       |
                                       |
                             (data)    V                    (process)
                             [ shape size, type info ] ---> ( generate_llm_output_func ) ---> "LLM Output"
                                       |                                                           |
                                       |                                                           |
 (int)                       (process) V                                                           V
 [ user prompt seed ] -----> ( generate_prompt_func ) ------> "User Prompt" ------> { Supervised Fine-Tuning of LLM }
"""


import pandas as pd
import numpy as np
import random
from collections import defaultdict

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common_values import CANVAS_WIDTH, CANVAS_HEIGHT, PROMPT_PREFIX, PROMPT_SUFFIX

MAX_PROMPT_SEED = 9999999
MAX_MODEL_STRUCTURE_SEED = 9999999
BACKGROUND_COLOR_LIST = [(255, 255, 255), (240, 240, 240), (224, 224, 224), (208, 208, 208),
                         (224, 224, 240), (192, 192, 224), (240, 232, 224), (224, 208, 192),
                         (224, 240, 232), (192, 224, 208), (232, 224, 240), (208, 192, 224)]
LINE_COLOR_LIST = [(0, 0, 0), (32, 32, 32), (64, 64, 64), (96, 96, 96),
                   (32, 32, 64), (64, 64, 128), (64, 48, 32), (128, 96, 64),
                   (32, 64, 48), (64, 128, 96), (48, 32, 64), (96, 64, 128)]
SUBTREE_PROB = 0.2


# For Deep Learning Model Prompt
user_prompt_part1_candidates = ['with', 'of', 'consist of']

input_node_names = ['input layer nodes', 'input nodes', 'input elements', 'input size']
hidden_layer_names = ['hidden layers', 'hiddens', 'intermediate layers', 'hidden layer', 'mid layers']
output_node_names = ['output layer nodes', 'output nodes', 'output elements', 'output size']


# For Flow Chart Prompt
NODE_GENERATE_STOP_AT = 15  # stop node generating when node count reaches this number

user_prompt_start = ['process that ',
                     'machine learning model that ',
                     'deep learning algorithm that ',
                     'Langchain process that ',
                     'RAG process that ',
                     'LLM process that ',
                     'data pre-processing algorithm that ',
                     'algorithm that ']
contain_marks = ['consists of ', 'contains, ', 'includes, ']

node_types = ['numeric', 'str', 'picture', 'db', 'chart', 'func', 'process', 'model']
node_types_cnt = len(node_types)

numeric_names = ['matrix', 'tensor', 'tensors', 'numeric values', 'matrices',
                 'buffer', 'buffers', 'numpy array', 'pytorch tensor', 'tensorflow tensor']
str_names = ['string', 'text', 'tokens', 'sentence', 'pandas dataframe']
picture_names = ['picture', 'figure', 'png file', 'jpg file']
db_names = ['DB', 'database', 'data storage', 'data store']
chart_names = ['chart', 'graph', 'table', 'line chart', 'histogram', 'experiment result']
func_names = ['function', 'code file', 'python file', 'python code']
process_names = ['process', 'python code', 'pre-processing', 'feature engineering', 'PCA', 'processing']
model_names = ['AI model', 'model', 'deep learning model', 'machine learning model', 'LLM', 'language model',
               'CNN model', 'neural network', 'NN']


# Layer Type, Layer Size 를 랜덤으로 결정
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - layer_config_seed (int) : 레이어 구성을 나타내는 int 값 (0 - 9,999,999)

# Returns:
# - layer_types (list(str)) : 딥러닝 모델의 각 레이어의 종류
# - layer_sizes (list(int)) : 딥러닝 모델의 각 레이어의 크기 (node 개수 or feature map 크기)

def generate_dl_model_structure(layer_config_seed):

    # Dense
    if layer_config_seed % 3 == 0:
        dense_config_seed = layer_config_seed // 3

        # input layer
        input_nodes = dense_config_seed % 5 + 2

        # hidden layer (dense)
        hidden_layers = (dense_config_seed // 5) % 3 + 1

        if hidden_layers == 1:
            hidden_nodes = [(dense_config_seed // (5 * 3)) % 4 + 1]

        elif hidden_layers == 2:
            hidden_nodes = [(dense_config_seed // (5 * 3)) % 6 + 3,
                            (dense_config_seed // (5 * 3 * 6)) % 6 + 3]

        else:  # hidden_layers == 3
            hidden_nodes = [(dense_config_seed // (5 * 3)) % 6 + 3,
                            (dense_config_seed // (5 * 3 * 6)) % 6 + 6,
                            (dense_config_seed // (5 * 3 * 6 * 6)) % 6 + 3]

        # output layer
        output_nodes = (layer_config_seed // (5 * 3 * 6 * 6 * 6)) % 2 + 1

        layer_types = ['dense'] * (hidden_layers + 2)
        layer_sizes = [input_nodes] + hidden_nodes + [output_nodes]

    # Conv. + Pool. + Dense (Convolutional Neural Network)
    else:
        conv_config_seed = layer_config_seed // 3

        # input layer
        input_sizes = [28, 32, 64, 128, 224, 256, 512, 768]
        conv_pool_times = [2, 2, 3, 4, 5, 5, 6, 7]

        input_size = input_sizes[conv_config_seed % 8]
        conv_pool_cnt = conv_pool_times[conv_config_seed % 8]
        consecutive_conv_layers = [(2 if random.random() < 0.3 else 1) for _ in range(conv_pool_cnt)]

        # size after Conv. and Pool. layers
        conv_pool_size = []
        conv_pool_layer_type = []
        current_size = input_size

        for c in range(conv_pool_cnt):

            # conv. layers
            for cc in range(consecutive_conv_layers[c]):
                current_size -= 2
                conv_pool_size.append(current_size)
                conv_pool_layer_type.append('conv')

            # pooling layer
            current_size = current_size // 2
            conv_pool_size.append(current_size)
            conv_pool_layer_type.append('pool')

        # hidden layer (dense after Conv. + Pool, fully-connected)
        hidden_layers = (conv_config_seed // 8) % 3 + 1

        if hidden_layers == 1:
            hidden_node_counts = [[64], [128], [256], [512], [1024]]
            hidden_nodes = hidden_node_counts[(conv_config_seed // (8 * 3)) % 5]

        elif hidden_layers == 2:
            hidden_node_counts = [[256, 32], [512, 64], [512, 128], [1024, 64], [1024, 128], [1024, 256]]
            hidden_nodes = hidden_node_counts[(conv_config_seed // (8 * 3)) % 6]

        else:  # hidden_layers == 3
            hidden_node_counts = [[256, 64, 16], [512, 128, 16], [512, 128, 32], [512, 256, 64], [1024, 256, 64]]
            hidden_nodes = hidden_node_counts[(conv_config_seed // (8 * 3)) % 5]

        # output layer
        output_nodes = (layer_config_seed // (8 * 3 * 5 * 6)) % 2 + 1

        layer_types = ['cnn_input'] + conv_pool_layer_type + ['dense'] * (hidden_layers + 1)
        layer_sizes = [input_size] + conv_pool_size + hidden_nodes + [output_nodes]

    return layer_types, layer_sizes


# Deep Learning 모델 구조 관련, layer type 별 사용자 입력 프롬프트에 추가할 부분 생성 (Conv. + Pool. 그룹의 모든 각 레이어 체크 시)
# Create Date : 2025.03.19
# Last Update Date : -

# Arguments:
# - layer_type          (str) : 딥러닝 모델의 해당 레이어의 종류
# - layer_size          (int) : 딥러닝 모델의 해당 레이어의 크기 (node 개수 or feature map 크기)
# - conv                (str) : Conv. layer 를 나타내는 단어
# - pooling_type        (str) : Pooling layer 의 type (max, average) 을 나타내는 단어
# - additional_then     (str) : 확률적으로 추가되는 'then ' 접속사 (또는 empty string)
# - additional_and_then (str) : 확률적으로 추가되는 'and ', 'then ' 또는 'and then ' 접속사 (또는 empty string)

# Returns:
# - added_user_prompt (str) : 사용자 입력 프롬프트에 추가할 부분

def get_dl_model_prompt_not_conv_pool_at_once(layer_type, layer_size, conv, pooling_type,
                                              additional_then, additional_and_then):
    r = random.random()

    # Convolutional Layer
    if layer_type == 'conv':
        if r < 0.2:
            return f'{conv} layer, {additional_and_then}'
        elif r < 0.35:
            return f'3 x 3 {conv} layer, {additional_and_then}'
        elif r < 0.5:
            return f'3x3 {conv} layer, {additional_and_then}'
        elif r < 0.65:
            return f'3 * 3 {conv} layer, {additional_and_then}'
        elif r < 0.8:
            return f'3*3 {conv} layer, {additional_and_then}'
        elif r < 0.9:
            return f'{conv} layer (output is {layer_size} x {layer_size}), {additional_and_then}'
        else:
            return f'{conv} layer (output: {layer_size} x {layer_size} feature map), {additional_and_then}'

    # Pooling Layer
    else:
        if r < 0.3:
            return f'{pooling_type}pooling layer, {additional_then}'
        elif r < 0.475:
            return f'2 x 2 {pooling_type}pooling layer, {additional_then}'
        elif r < 0.65:
            return f'2x2 {pooling_type}pooling layer, {additional_then}'
        elif r < 0.825:
            return f'2 * 2 {pooling_type}pooling layer, {additional_then}'
        else:
            return f'2*2 {pooling_type}pooling layer, {additional_then}'


# Deep Learning 모델 구조 관련, layer type 별 사용자 입력 프롬프트에 추가할 부분 생성
# Create Date : 2025.03.19
# Last Update Date : -

# Arguments:
# - layer_idx              (int)       : 해당 레이어의 인덱스 (모든 레이어 중)
# - layer_type             (str)       : 딥러닝 모델의 해당 레이어의 종류
# - layer_size             (int)       : 딥러닝 모델의 해당 레이어의 크기 (node 개수 or feature map 크기)
# - layer_types            (list(str)) : 딥러닝 모델의 모든 레이어의 종류
# - last_pooling_layer_idx (int)       : 딥러닝 모델 전체를 기준으로, 마지막 pooling layer 의 인덱스
# - prompt_seed            (int)       : 프롬프트 형식을 나타내는 int 값 (0 - 9,999,999)
# - say_conv_pool_at_once  (bool)      : Conv. + Pool. 그룹의 마지막 레이어인 Pooling Layer 에서만 레이어 개수 체크할지 여부

# Returns:
# - added_user_prompt       (str) : layer type 별 사용자 입력 프롬프트에 추가할 부분
# - last_pooling_layer_idx_ (int) : last_pooling_layer_idx 의 업데이트된 값

def get_dl_model_prompt_of_layer_type(layer_idx, layer_type, layer_size, layer_types,
                                      prompt_seed, last_pooling_layer_idx, say_conv_pool_at_once):

    added_user_prompt = ''
    last_pooling_layer_idx_ = last_pooling_layer_idx

    additional_and = "and " if random.random() < 0.5 else ""
    additional_then = "then " if random.random() < 0.5 else ""
    additional_and_then = additional_and + additional_then

    # Conv. or Pool. layer
    if layer_type in ['conv', 'pool']:
        conv = "convolutional" if random.random() < 0.5 else "conv"

        if random.random() < 0.4:
            pooling_type = "max " if random.random() < 0.5 else "average "
        else:
            pooling_type = ""

        # Conv. + Pool. 그룹의 마지막 레이어인 Pooling Layer 에서만 해당 그룹의 종류 별 레이어 개수 체크
        if say_conv_pool_at_once:
            if layer_type == 'conv':
                return '', last_pooling_layer_idx

            current_conv_and_pool_group = layer_types[last_pooling_layer_idx_ : layer_idx + 1]
            conv_layers = current_conv_and_pool_group.count('conv')
            last_pooling_layer_idx_ = layer_idx

            if conv_layers > 1:
                added_user_prompt += f'{conv_layers} 3 x 3 {conv} layers and a 2 x 2 {pooling_type}pooling layer, {additional_then}'
            else:
                added_user_prompt += f'a 3 x 3 {conv} layer and a 2 x 2 {pooling_type}pooling layer, {additional_then}'

        else:
            added_user_prompt += get_dl_model_prompt_not_conv_pool_at_once(layer_type, layer_size, conv, pooling_type,
                                                                           additional_then, additional_and_then)

    # CNN input layer
    elif layer_type == 'cnn_input':
        additional_img = ' image' if random.random() < 0.5 else ''
        img_size = f'{layer_size} * {layer_size}' if random.random() < 0.5 else f'{layer_size} x {layer_size}'
        added_user_prompt += f'{img_size} input{additional_img}, '

    # input layer (dense)
    elif layer_idx == 0:
        input_node_name = input_node_names[(prompt_seed // (6 * 3)) % 4]
        added_user_prompt += f'{layer_size} {input_node_name}, '

    # output layer (dense)
    elif layer_idx == len(layer_types) - 1:
        output_node_name = output_node_names[(prompt_seed // (6 * 3 * 4 * 5)) % 4]
        added_user_prompt += f'and {layer_size} {output_node_name} '

    # last hidden layer (dense)
    elif layer_idx == len(layer_types) - 2:
        hidden_layer_name = hidden_layer_names[(prompt_seed // (6 * 3 * 4)) % 5]

        if layer_idx == 1 or layer_types[layer_idx - 1] in ['conv', 'pool']:  # only one hidden layer
            added_user_prompt += f'and {layer_size} nodes in {hidden_layer_name}, '

        else:
            r = random.random()
            additional_and = "and " if random.random() < 0.5 else ""
            hidden_layer_cnt = (len(layer_types) - 2) - (layer_types.count('conv') + layer_types.count('pool'))

            if r < 0.25:
                added_user_prompt += f'{additional_and}{layer_size} nodes in each of the {hidden_layer_cnt} {hidden_layer_name}, '
            elif r < 0.5:
                added_user_prompt += f'{additional_and}{layer_size} nodes in the {hidden_layer_name}, '
            elif r < 0.75:
                added_user_prompt += f'{additional_and}{layer_size} nodes in {hidden_layer_cnt} {hidden_layer_name}, '
            else:
                added_user_prompt += f'{additional_and}{layer_size} nodes in {hidden_layer_name}, '

    # hidden layer (dense)
    else:
        if random.random() < 0.5:
            added_user_prompt += f'{layer_size}, '
        else:
            added_user_prompt += f'{layer_size} '

    return added_user_prompt, last_pooling_layer_idx_


# Deep Learning 모델 구조 관련 사용자 입력 프롬프트 생성
# Create Date : 2025.03.17
# Last Update Date : 2025.03.19
# - layer type 에 따른 프롬프트 생성을 별도 함수 (get_dl_model_prompt_of_layer_type) 로 분리

# Arguments:
# - prompt_seed (int)       : 프롬프트 형식을 나타내는 int 값 (0 - 9,999,999)
# - layer_types (list(str)) : 딥러닝 모델의 각 레이어의 종류
# - layer_sizes (list(int)) : 딥러닝 모델의 각 레이어의 크기 (node 개수 or feature map 크기)

# Returns:
# - entire_prompt (str) : Deep Learning 모델 구조 관련 학습 데이터셋의 입력 프롬프트
# - user_prompt   (str) : Prompt Engineering 을 위한 앞뒤 부분을 제외한 순수 유저 프롬프트

def generate_dl_model_prompt(prompt_seed, layer_types, layer_sizes):
    is_cnn = (layer_types[0] == 'cnn_input')

    if is_cnn:
        user_prompt_part0_candidates = ['A deep learning model', 'DL model', 'neural network',
                                        'NN', 'NN model', 'neural net']
    else:
        user_prompt_part0_candidates = ['Convolutional neural network', 'Conv neural network',
                                        'DL model', 'neural network', 'CNN', 'CNN model']

    user_prompt = (user_prompt_part0_candidates[prompt_seed % 6] + ' ' +
                   user_prompt_part1_candidates[(prompt_seed // 6) % 3] + ' ')

    # 한 group (1~2 Conv. + 1 Pool.) 내의 모든 Conv. 레이어를 묶어서 한번에 표현
    say_conv_pool_at_once = random.random() < 0.4
    last_pooling_layer_idx = 0

    for idx, (t, s) in enumerate(zip(layer_types, layer_sizes)):

        added_user_prompt, updated_last_pool_idx = get_dl_model_prompt_of_layer_type(layer_idx=idx,
                                                                                     layer_type=t,
                                                                                     layer_size=s,
                                                                                     layer_types=layer_types,
                                                                                     prompt_seed=prompt_seed,
                                                                                     last_pooling_layer_idx=last_pooling_layer_idx,
                                                                                     say_conv_pool_at_once=say_conv_pool_at_once)

        user_prompt += added_user_prompt
        last_pooling_layer_idx = updated_last_pool_idx

    entire_prompt = PROMPT_PREFIX + user_prompt + PROMPT_SUFFIX
    return entire_prompt, user_prompt


# Deep Learning 모델 구조 관련 레이어 별 출력값을 생성하기 위한 shape 정보 계산
# Create Date : 2025.03.17
# Last Update Date : 2025.03.18
# - 각 layer 안에 속한 node 의 위치 및 가로/세로 길이 버그 수정

# Arguments:
# - layer_size        (int)  : 해당 레이어의 크기 (node 개수 or feature map 크기)
# - layer_idx         (int)  : 해당 레이어의 인덱스 (모든 레이어 중)
# - layer_cnt         (int)  : 모든 레이어의 개수
# - max_layer_size    (int)  : 모든 레이어의 크기의 최댓값
# - diagram_direction (str)  : 다이어그램 그리기 방향 (가로, 세로)
# - draw_each_node    (bool) : 각 레이어의 node (Conv., Pool. 제외) 를 표시할 것인가?

# Returns:
# - shapes_info (list(dict)) : shape 정보 리스트,
#                              각 항목은 {'x': x (center), 'y': y (center), 'w': width, 'h': height}

def generate_shapes_info_of_layer(layer_size, layer_idx, layer_cnt, diagram_direction, draw_each_node, max_layer_size):

    shapes_info = []
    layer_lengths_relative = np.log(layer_size + 1) / np.log(max_layer_size + 1)

    if diagram_direction == 'horizontal':
        layer_x = (layer_idx + 1) / (layer_cnt + 1) * CANVAS_WIDTH
        layer_y = CANVAS_HEIGHT // 2
        layer_width = CANVAS_WIDTH // (2 * layer_cnt)
        layer_height = int(0.6 * CANVAS_HEIGHT * layer_lengths_relative)

        if draw_each_node:
            node_xs = [layer_x for _ in range(layer_size)]
            node_ys = np.linspace(CANVAS_HEIGHT // 2 - layer_height // 2,
                                  CANVAS_HEIGHT // 2 + layer_height // 2,
                                  layer_size + 2)[1:-1]
            node_widths = [layer_width for _ in range(layer_size)]
            node_heights = [layer_height // (2 * layer_size) for _ in range(layer_size)]

            for x, y, w, h in zip(node_xs, node_ys, node_widths, node_heights):
                shapes_info.append({'x': x, 'y': y, 'w': w, 'h': h})

    else:  # vertical
        layer_x = CANVAS_WIDTH // 2
        layer_y = (layer_idx + 1) / (layer_cnt + 1) * CANVAS_HEIGHT
        layer_width = int(0.75 * CANVAS_WIDTH * layer_lengths_relative)
        layer_height = CANVAS_HEIGHT // (2 * layer_cnt)

        if draw_each_node:
            node_xs = np.linspace(CANVAS_WIDTH // 2 - layer_width // 2,
                                  CANVAS_WIDTH // 2 + layer_width // 2,
                                  layer_size + 2)[1:-1]
            node_ys = [layer_y for _ in range(layer_size)]
            node_widths = [layer_width // (2 * layer_size) for _ in range(layer_size)]
            node_heights = [layer_height for _ in range(layer_size)]

            for x, y, w, h in zip(node_xs, node_ys, node_widths, node_heights):
                shapes_info.append({'x': x, 'y': y, 'w': w, 'h': h})

    if not draw_each_node:
        shapes_info.append({'x': layer_x,
                            'y': layer_y,
                            'w': layer_width,
                            'h': layer_height})

    return shapes_info


# Deep Learning 모델 구조 관련 레이어 별 shapes 정보를 이용하여 각 레이어에 대한 출력값을 생성
# Create Date : 2025.03.18
# Last Update Date : -

# Arguments:
# - layer_idx          (int)        : 출력값을 생성할 레이어의 번호 (index)
# - layer_property     (dict)       : 해당 레이어에 대한 속성,
#                                     {'back_color': 배경색, 'line_color': 연결선 색, 'shape': 도형 모양, 'line_shape': 연결선 모양}
# - shapes_info        (list(dict)) : shape 정보의 리스트,
#                                     각 항목은 {'x': x (center), 'y': y (center), 'w': width, 'h': height}
# - node_cnt_per_layer (list)       : 각 layer 별 필요한 도형 (node) 의 개수 (Diagram 의 Shape ID 용)

# Returns:
# - model_output_of_layer (str) : 해당 layer 에 대한 다이어그램 형식의 텍스트 (draw_diagram/diagram.txt 참고)

def generate_dl_model_llm_output_of_layer(layer_idx, layer_property, shapes_info, node_cnt_per_layer):

    model_output_of_layer = ''
    is_last_layer = (layer_idx == len(node_cnt_per_layer) - 1)

    previous_all_layer_nodes_cnt = sum(node_cnt_per_layer[:layer_idx])
    cur_layer_node_cnt = node_cnt_per_layer[layer_idx]
    if is_last_layer:
        next_layer_node_cnt = 0
    else:
        next_layer_node_cnt = node_cnt_per_layer[layer_idx + 1]

    for i in range(cur_layer_node_cnt):
        shape_id = previous_all_layer_nodes_cnt + i

        center_x = int(shapes_info[i]['x'])
        center_y = int(shapes_info[i]['y'])
        shape = layer_property['shape']
        width = int(shapes_info[i]['w'])
        height = int(shapes_info[i]['h'])
        line = layer_property['line_shape']

        back_color = layer_property['back_color']
        line_color = layer_property['line_color']

        if is_last_layer:
            connected_nodes = []
        else:
            connected_nodes = list(range(previous_all_layer_nodes_cnt + cur_layer_node_cnt,
                                         previous_all_layer_nodes_cnt + cur_layer_node_cnt + next_layer_node_cnt))

        model_output_of_layer_list = [shape_id, center_x, center_y, shape, width, height, line,
                                      back_color, line_color, connected_nodes]

        model_output_of_layer += str(model_output_of_layer_list) + '\n'

    return model_output_of_layer


# Deep Learning 모델 구조 관련 LLM 출력값 생성
# Create Date : 2025.03.18
# Last Update Date : -

# Arguments:
# - layer_types (list(str)) : 딥러닝 모델의 각 레이어의 종류
# - layer_sizes (list(int)) : 딥러닝 모델의 각 레이어의 크기 (node 개수 or feature map 크기)

# Returns:
# - model_output (str) : 다이어그램 형식의 텍스트 (draw_diagram/diagram.txt 참고)

def generate_dl_model_llm_output(layer_types, layer_sizes):

    model_output = ''
    shapes_info_dict = {}

    n = len(layer_types)
    diagram_direction = 'horizontal' if n > 8 else 'vertical'
    is_cnn = (layer_types[0] == 'cnn_input')
    dense_weight_matrix_elements = [layer_sizes[i] * layer_sizes[i + 1] for i in range(n - 1)]
    draw_each_node = not is_cnn and max(dense_weight_matrix_elements) <= 45

    back_colors = random.choices(BACKGROUND_COLOR_LIST, k=6)
    line_colors = random.choices(LINE_COLOR_LIST, k=6)
    shape = random.choices(['rectangle', 'round rectangle'], k=6)
    line = random.choices(['solid arrow', 'solid line'], k=1)

    property_each_layer_type = {'cnn_input': {'back_color': back_colors[0],
                                              'line_color': line_colors[0],
                                              'shape': shape[0],
                                              'line_shape': line[0]},

                                'conv': {'back_color': back_colors[1],
                                         'line_color': line_colors[1],
                                         'shape': shape[1],
                                         'line_shape': line[0]},

                                'pool': {'back_color': back_colors[2],
                                         'line_color': line_colors[2],
                                         'shape': shape[2],
                                         'line_shape': line[0]},

                                'dense_input': {'back_color': back_colors[3],
                                                'line_color': line_colors[3],
                                                'shape': 'circle' if draw_each_node else shape[3],
                                                'line_shape': 'solid arrow' if draw_each_node else line[0]},

                                'dense_hidden': {'back_color': back_colors[4],
                                                 'line_color': line_colors[4],
                                                 'shape': 'circle' if draw_each_node else shape[4],
                                                 'line_shape': 'solid arrow' if draw_each_node else line[0]},

                                'dense_output': {'back_color': back_colors[5],
                                                 'line_color': line_colors[5],
                                                 'shape': 'circle' if draw_each_node else shape[5],
                                                 'line_shape': 'solid arrow' if draw_each_node else line[0]}}

    # generate shape info for each layer
    node_cnt_per_layer = []

    for layer_idx, (layer_type, layer_size) in enumerate(zip(layer_types, layer_sizes)):
        shapes_info = generate_shapes_info_of_layer(layer_size=layer_size,
                                                    layer_idx=layer_idx,
                                                    layer_cnt=n,
                                                    diagram_direction=diagram_direction,
                                                    draw_each_node=draw_each_node,
                                                    max_layer_size=max(layer_sizes))
        shapes_info_dict[layer_idx] = shapes_info
        node_cnt_per_layer.append(len(shapes_info))

    # generate output for each layer
    for layer_idx, (layer_type, layer_size) in enumerate(zip(layer_types, layer_sizes)):
        if layer_type == 'dense':
            if layer_idx == 0:
                layer_property = property_each_layer_type['dense_input']
            elif layer_idx == n - 1:
                layer_property = property_each_layer_type['dense_output']
            else:
                layer_property = property_each_layer_type['dense_hidden']

        else:
            layer_property = property_each_layer_type[layer_type]

        model_output_of_layer = generate_dl_model_llm_output_of_layer(layer_idx,
                                                                      layer_property,
                                                                      shapes_info=shapes_info_dict[layer_idx],
                                                                      node_cnt_per_layer=node_cnt_per_layer)
        model_output += model_output_of_layer

    return model_output


# Flow Chart 의 도형 Size, Type 를 랜덤으로 결정
# Create Date : 2025.03.18
# Last Update Date : 2025.03.20
# - process/func 의 인접 노드에 process/func 이 있을 수 있는 버그 해결
# - 도형이 나타낼 수 있는 요소 유형 중 'model' 추가에 따른 수정

# Arguments:
# - shape_config_seed (int) : 도형 구성을 나타내는 int 값 (0 - 9,999,999)

# Returns:
# - shape_types (list(dict)) : 각 도형의 종류
#                              [{'id': int, 'type': str, 'connected_node_ids': list(str)},
#                               {'id': int, 'type': str, 'connected_node_ids': list(str)}, ...] 형식
# - shape_sizes (list(int))  : 각 도형의 크기

def generate_flow_chart_structure(shape_config_seed):
    node_info = {}
    max_depth_from_start = shape_config_seed % 8 + 3

    # 현재 node 의 type 에 따라, type 가 아직 정해지지 않은 인접한 node 의 type 결정
    def decide_adjacent_node_type(node_type):
        if node_type in ['func', 'process', 'model']:
            r = random.randint(0, 4)
            return node_types[r]
        else:
            if random.random() < 0.25:
                r = random.randint(0, 4)
                return node_types[r]
            else:
                r = random.random()
                if r < 0.2:
                    return 'func'
                elif r < 0.5:
                    return 'process'
                else:
                    return 'model'

    first_node_type = node_types[(shape_config_seed // 8) % 6]
    first_node = {'id': 0, 'type': first_node_type, 'depth': 0}

    # generate first (max_depths + 1) consecutive nodes
    for i in range(max_depth_from_start + 1):
        if i == 0:
            node_info[i] = {'type': first_node_type,
                            'connected_node_ids': [1]}

        elif i < max_depth_from_start:
            node_info[i] = {'type': decide_adjacent_node_type(node_info[i - 1]['type']),
                            'connected_node_ids': [i + 1]}

        else:
            node_info[i] = {'type': decide_adjacent_node_type(node_info[i - 1]['type']),
                            'connected_node_ids': []}

    # initialize DFS stack to generate nodes
    dfs_stack = [first_node]
    for i in range(max_depth_from_start):
        dfs_stack.append({'id': i, 'type': node_info.get(i)['type'], 'depth': i})
    dfs_stack = dfs_stack[::-1]

    # generate nodes
    while len(dfs_stack) > 0:
        current_node = dfs_stack.pop(-1)

        # add subtrees
        if current_node['depth'] < max_depth_from_start:
            while random.random() < SUBTREE_PROB:
                current_node_id = current_node['id']
                current_node_type = current_node['type']
                new_node_id = len(node_info)
                new_node_type = decide_adjacent_node_type(current_node_type)

                node_info[current_node_id]['connected_node_ids'].append(new_node_id)
                node_info[new_node_id] = {'type': new_node_type, 'connected_node_ids': []}

                dfs_stack.append({'id': new_node_id, 'type': new_node_type, 'depth': current_node['depth'] + 1})

        if len(node_info) >= NODE_GENERATE_STOP_AT:
            break

        # add inverse-subtrees (incoming node)
        if current_node['depth'] > 0:
            while random.random() < SUBTREE_PROB:
                current_node_id = current_node['id']
                current_node_type = current_node['type']
                new_node_id = len(node_info)
                new_node_type = decide_adjacent_node_type(current_node_type)

                node_info[new_node_id] = {'type': new_node_type, 'connected_node_ids': [current_node_id]}

                dfs_stack.append({'id': new_node_id, 'type': new_node_type, 'depth': current_node['depth'] - 1})

        if len(node_info) >= NODE_GENERATE_STOP_AT:
            break

    # shape_types, shape_sizes 지정
    shape_types = []

    for node_id, node_property in node_info.items():
        shape_types.append({'id': node_id,
                            'type': node_property['type'],
                            'connected_node_ids': node_property['connected_node_ids']})

    shape_types.sort(key=lambda x: x['id'])

    shape_sizes = []
    for shape_info in shape_types:
        if shape_info['type'] in ['numeric', 'str']:
            shape_sizes.append(0.6)
        elif shape_info['type'] in ['picture', 'db', 'chart']:
            shape_sizes.append(1.0)
        else:
            shape_sizes.append(0.3)

    return shape_types, shape_sizes


# Flow Chart 구조 중 Processing 을 나타내는 node 에 대한 사용자 입력 프롬프트의 부분 생성
# Create Date : 2025.03.19
# Last Update Date : -

# Arguments:
# - node_name            (str)       : Processing node 를 나타내는 이름
# - incoming_node_names  (list(str)) : 해당 node 이전 단계를 나타내는 node 들을 나타내는 이름
# - connected_node_names (list(str)) : 해당 node 이후 단계를 나타내는 node 들을 나타내는 이름

# Returns:
# - process_user_prompt (str) : 해당 Processing 을 나타내는 node 에 대한 사용자 입력 프롬프트의 부분

def get_process_user_prompt(node_name, incoming_node_names, connected_node_names):
    process_user_prompt = ''

    if random.random() < 0.5:
        additional_and = " and " if random.random() < 0.5 else ", "
        process_or_handle = random.choice(['inputs', 'handle', 'process'])

        process_user_prompt += f'{node_name} that '

        if random.random() < 0.5:
            if len(incoming_node_names) > 0:
                process_user_prompt += f'{process_or_handle} {additional_and.join(incoming_node_names)}'
                if len(connected_node_names) > 0:
                    process_user_prompt += f', and outputs {additional_and.join(connected_node_names)}'

            elif len(connected_node_names) > 0:
                process_user_prompt += f'outputs {additional_and.join(connected_node_names)}'

        else:
            if len(incoming_node_names) > 0:
                process_user_prompt += f', with {additional_and.join(incoming_node_names)} as input'
                if len(connected_node_names) > 0:
                    process_user_prompt += f', and {additional_and.join(connected_node_names)} as output'

            elif len(connected_node_names) > 0:
                process_user_prompt += f'with {additional_and.join(connected_node_names)} as output'

    else:
        and_mark = ' and ' if random.random() < 0.5 else ', '

        if len(incoming_node_names) > 0:
            process_user_prompt += f'inputs {and_mark.join(incoming_node_names)}'
            if len(connected_node_names) > 0:
                process_user_prompt += f', and outputs {and_mark.join(connected_node_names)}'

            it_or_them = 'it' if len(incoming_node_names) + len(connected_node_names) == 1 else 'them'
            process_user_prompt += f' and process {it_or_them} with {node_name}'

        elif len(connected_node_names) > 0:
            process_user_prompt += f'outputs {and_mark.join(connected_node_names)}'

            it_or_them = 'it' if len(connected_node_names) == 1 else 'them'
            process_user_prompt += f' and process {it_or_them} with {node_name}'

        else:
            process_user_prompt += f' process {node_name}'

    return process_user_prompt


# Flow Chart 구조 관련 사용자 입력 프롬프트 생성
# Create Date : 2025.03.19
# Last Update Date : 2025.03.20
# - 도형이 나타낼 수 있는 요소 유형 중 'model' 추가에 따른 수정

# Arguments:
# - prompt_seed (int)        : 프롬프트 형식을 나타내는 int 값 (0 - 9,999,999)
# - shape_types (list(dict)) : 각 도형의 종류
#                              [{'id': int, 'type': str, 'connected_node_ids': list(str)},
#                               {'id': int, 'type': str, 'connected_node_ids': list(str)}, ...] 형식
# - shape_sizes (list(int))  : 각 도형의 크기

# Returns:
# - entire_prompt (str) : Flow Chart 구조 관련 학습 데이터셋의 입력 프롬프트
# - user_prompt   (str) : Prompt Engineering 을 위한 앞뒤 부분을 제외한 순수 유저 프롬프트

def generate_flow_chart_prompt(prompt_seed, shape_types, shape_sizes):
    n = len(shape_types)

    user_prompt = random.choice(user_prompt_start) + random.choice(contain_marks)

    default_numeric_name = numeric_names[prompt_seed % 10]
    default_str_name = str_names[(prompt_seed // 10) % 5]
    default_picture_name = picture_names[(prompt_seed // (10 * 5)) % 4]
    default_db_name = db_names[(prompt_seed // (10 * 5 * 4)) % 4]
    default_chart_name = chart_names[(prompt_seed // (10 * 5 * 4 * 4)) % 6]
    default_func_name = func_names[(prompt_seed // (10 * 5 * 4 * 4 * 6)) % 4]
    default_process_name = process_names[(prompt_seed // (10 * 5 * 4 * 4 * 6 * 4)) % 6]
    default_model_name = model_names[(prompt_seed // (10 * 5 * 4 * 4 * 6 * 4 * 6)) % 9]

    # divide parts of each node by new-line
    divide_by_newline = random.random() < 0.75
    numbering_mark = '-' if random.random() < 0.5 else '*'
    if divide_by_newline:
        user_prompt += f'\n{numbering_mark} '
    else:
        additional_first = 'first, ' if random.random() < 0.5 else ''
        user_prompt += additional_first

    # node 의 종류에 따라 이름 반환
    def get_node_name(node_type_str):
        if node_type_str == 'numeric':
            return default_numeric_name if random.random() < 0.5 else random.choice(numeric_names)
        elif node_type_str == 'str':
            return default_str_name if random.random() < 0.5 else random.choice(str_names)
        elif node_type_str == 'picture':
            return default_picture_name if random.random() < 0.5 else random.choice(picture_names)
        elif node_type_str == 'db':
            return default_db_name if random.random() < 0.5 else random.choice(db_names)
        elif node_type_str == 'chart':
            return default_chart_name if random.random() < 0.5 else random.choice(chart_names)
        elif node_type_str == 'func':
            return default_func_name if random.random() < 0.5 else random.choice(func_names)
        elif node_type_str == 'process':
            return default_process_name if random.random() < 0.5 else random.choice(process_names)
        else:  # model
            return default_model_name if random.random() < 0.5 else random.choice(model_names)

    # phrase between each node
    def get_phrase_between_each_node(idx):
        if idx < n - 1:
            if divide_by_newline:
                return f'\n{numbering_mark} '
            else:
                additional_then = "then " if random.random() < 0.5 else ""
                return f', and {additional_then}'
        else:
            return '.'

    # add incoming node info
    for i in range(n):
        shape_types[i]['incoming_node_ids'] = []

    for i in range(n):
        for connected_node_id in shape_types[i]['connected_node_ids']:
            shape_types[connected_node_id]['incoming_node_ids'].append(i)

    # generate user prompt
    for i in range(n):
        node_type = shape_types[i]['type']
        node_name = get_node_name(node_type)

        incoming_nodes = shape_types[i]['incoming_node_ids']
        incoming_node_types = [shape_types[node_id]['type'] for node_id in incoming_nodes]
        incoming_node_names = [get_node_name(shape_types[node_id]['type']) for node_id in incoming_nodes]

        connected_nodes = shape_types[i]['connected_node_ids']
        connected_node_names = [get_node_name(shape_types[node_id]['type']) for node_id in connected_nodes]

        # 해당 node 가 처리 프로세스 (function, process, model) 인 경우
        if node_type in ['func', 'process', 'model']:
            user_prompt += get_process_user_prompt(node_name, incoming_node_names, connected_node_names)
            user_prompt += get_phrase_between_each_node(i)

        # 해당 node 가 데이터이면서 앞의 node 도 모두 데이터인 경우
        elif (node_type not in ['func', 'process', 'model'] and len(incoming_node_names) > 0 and
              incoming_node_types.count('func') == 0 and incoming_node_types.count('process') == 0 and
              incoming_node_types.count('model') == 0):

            process_name = get_node_name('process')
            user_prompt += f'a {process_name} converts ' + ' and '.join(incoming_node_names) + ' '
            user_prompt += f'into {node_name}'

            user_prompt += get_phrase_between_each_node(i)

    # remove duplicated spaces and unnecessary marks
    user_prompt = user_prompt.replace(' , ', ', ').replace('  ', ' ')
    for unn_mark in ['and then ', 'and ', '- ', '* ', ', \n']:
        if user_prompt.endswith(unn_mark):
            user_prompt = user_prompt[:-len(unn_mark)] + '\n'

    entire_prompt = PROMPT_PREFIX + user_prompt + PROMPT_SUFFIX
    return entire_prompt, user_prompt


# 현재 node 및 현재 node 와 forward 방향으로 연결된 모든 node 에 대해, data 만 있는지 (process, function, model 이 없는지) 확인
# Create Date : 2025.03.19
# Last Update Date : 2025.03.20
# - 도형이 나타낼 수 있는 요소 유형 중 'model' 추가에 따른 수정

# Arguments:
# - shape_info (list(dict)) : 각 도형의 종류 및 각종 정보
#                             [{'id': int, 'type': str, 'connected_node_ids': list(str), 'depth': int},
#                              {'id': int, 'type': str, 'connected_node_ids': list(str), 'depth': int}, ...] 형식
# - node_id    (int)        : 확인을 원하는 도형의 node id

# Returns:
# - is_all_data (bool) : 해당 node 및 모든 connected node 에 대해, data 만 있는지 (process, function, model 이 없는지) 의 여부

def is_this_and_all_connected_nodes_data(shape_info, node_id):
    if shape_info[node_id]['type'] in ['func', 'process', 'model']:
        return False

    connected_node_ids = shape_info[node_id]['connected_node_ids']
    for connected_node_id in connected_node_ids:
        if shape_info[connected_node_id]['type'] in ['func', 'process', 'model']:
            return False

    return True


# Flow Chart 구조 관련 LLM 출력값 생성 (다이어그램을 만들기 위한 데이터)
# Create Date : 2025.03.19
# Last Update Date : 2025.03.20
# - DB, model 의 경우에는 circle 로 표시하도록 수정

# Arguments:
# - shape_info  (list(dict)) : 각 도형의 종류 및 각종 정보
#                              [{'id': int, 'type': str, 'connected_node_ids': list(str), 'depth': int},
#                               {'id': int, 'type': str, 'connected_node_ids': list(str), 'depth': int}, ...] 형식
# - shape_sizes (list(int)) : 각 도형의 크기

# Returns:
# - model_output (str) : 다이어그램 형식의 텍스트 (draw_diagram/diagram.txt 참고)

def generate_flow_chart_llm_data(shape_info, shape_sizes):
    max_depth = max(info['depth'] for info in shape_info)
    min_depth = min(info['depth'] for info in shape_info)
    diff = max_depth - min_depth

    # count nodes per depth
    nodes_count = defaultdict(lambda: {'count': 0, 'checked': 0})
    for si in shape_info:
        nodes_count[si['depth']]['count'] += 1
    max_count_by_depth = max(nodes_count[i]['count'] for i in nodes_count.keys())

    # horizontal / vertical
    diagram_direction = 'horizontal' if diff >= 6 else 'vertical'

    # property by node type
    back_colors = random.choices(BACKGROUND_COLOR_LIST, k=node_types_cnt)
    line_colors = random.choices(LINE_COLOR_LIST, k=node_types_cnt)
    shape = random.choices(['rectangle', 'round rectangle'], k=node_types_cnt)
    line = ['solid arrow'] * node_types_cnt

    property_each_node_type = {}
    for idx, node_type in enumerate(node_types):
        property_each_node_type[node_type] = {'back_color': back_colors[idx],
                                              'line_color': line_colors[idx],
                                              'shape': 'circle' if node_type in ['db', 'model'] else shape[idx],
                                              'line_shape': line[idx]}

    # write LLM model output
    model_output = ''

    for si, ss in zip(shape_info, shape_sizes):
        node_id = si['id']
        node_type = si['type']
        node_property = property_each_node_type[node_type]

        count = nodes_count[si['depth']]['count']
        checked = nodes_count[si['depth']]['checked']
        node_position = (checked + 0.5) - count / 2  # positive for right/bottom, 0 for center, negative for left/top
        nodes_count[si['depth']]['checked'] += 1

        if diagram_direction == 'horizontal':
            node_x = CANVAS_WIDTH * (0.5 + (si['depth'] - min_depth)) / (1.0 + diff)
            node_y = CANVAS_HEIGHT * (0.5 + node_position / max_count_by_depth)

        else:  # vertical
            node_x = CANVAS_WIDTH * (0.5 + node_position / max_count_by_depth)
            node_y = CANVAS_HEIGHT * (0.5 + (si['depth'] - min_depth)) / (1.0 + diff)

        node_shape = node_property['shape']

        node_size_base = 110 - 6 * max(max_count_by_depth, diff)
        node_width = node_size_base * ss
        node_height = node_size_base * ss
        back_color = node_property['back_color']

        if is_this_and_all_connected_nodes_data(shape_info, node_id):
            line_shape = 'dashed line'
        else:
            line_shape = node_property['line_shape']
        line_color = node_property['line_color']

        connected_node_ids = si['connected_node_ids']

        # represent as Python list and add (as string)
        node_representation_list = [node_id, int(node_x), int(node_y), node_shape,
                                    int(node_width), int(node_height), line_shape,
                                    back_color, line_color, connected_node_ids]

        model_output += f'{node_representation_list}\n'

    return model_output


# Flow Chart 구조 관련 LLM 출력값 생성
# Create Date : 2025.03.19
# Last Update Date : -

# Arguments:
# - shape_types (list(dict)) : 각 도형의 종류
#                              [{'id': int, 'type': str, 'connected_node_ids': list(str)},
#                               {'id': int, 'type': str, 'connected_node_ids': list(str)}, ...] 형식
# - shape_sizes (list(int)) : 각 도형의 크기

# Returns:
# - model_output (str) : 다이어그램 형식의 텍스트 (draw_diagram/diagram.txt 참고)

def generate_flow_chart_llm_output(shape_types, shape_sizes):
    visited = [False for _ in range(len(shape_types))]
    visited[0] = True

    current_node_id = 0
    current_depth = 0
    dfs_stack = []

    while True:
        shape_types[current_node_id]['depth'] = current_depth

        # find connected / incoming nodes
        connected_node_ids = shape_types[current_node_id]['connected_node_ids']
        for cno_id in connected_node_ids:
            if not visited[cno_id]:
                dfs_stack.append({'id': cno_id, 'depth': current_depth + 1})
                visited[cno_id] = True

        incoming_node_ids = shape_types[current_node_id]['incoming_node_ids']
        for ino_id in incoming_node_ids:
            if not visited[ino_id]:
                dfs_stack.append({'id': ino_id, 'depth': current_depth - 1})
                visited[ino_id] = True

        if len(dfs_stack) == 0:
            break

        # go to next node
        next_node_info = dfs_stack.pop(-1)
        current_node_id = next_node_info['id']
        current_depth = next_node_info['depth']

    # generate diagram output
    model_output = generate_flow_chart_llm_data(shape_info=shape_types, shape_sizes=shape_sizes)

    return model_output


# LLM 학습 데이터셋 생성
# Create Date : 2025.03.18
# Last Update Date : 2025.03.21
# - DataFrame 에 task name 및 LLM 을 통해 생성해야 할 shape 의 정보 관련 항목 추가

# Arguments:
# - task_name                (str)  : DL Model or Flow-Chart 생성 task 의 이름 ('dl_model' or 'flowchart')
# - dataset_size             (int)  : 데이터셋 규모
# - generate_structure_func  (func) : shape type 및 shape size 등 다이어그램의 도형 정보 데이터 생성 함수
# - generate_prompt_func     (func) : 도형 정보 데이터를 이용하여 User Prompt 를 생성하는 함수
# - generate_llm_output_func (func) : User Prompt 에 대한 적절한 LLM Output (SFT 용) 생성하는 함수

# Returns:
# - dl_dataset (Pandas DataFrame) : Deep Learning 모델 구조 관련 학습 데이터셋

def generate_dataset(task_name, dataset_size, generate_structure_func, generate_prompt_func, generate_llm_output_func):
    inputs = []
    outputs = []
    user_prompts = []
    dest_shape_info = []

    # 데이터셋 생성
    for i in range(dataset_size):

        # generate user prompt (LLM input)
        shape_types, shape_sizes = generate_structure_func(random.randint(0, MAX_MODEL_STRUCTURE_SEED))
        entire_prompt, user_prompt = generate_prompt_func(random.randint(0, MAX_PROMPT_SEED), shape_types, shape_sizes)
        inputs.append(entire_prompt)
        user_prompts.append(user_prompt)

        # generate LLM output for training
        llm_output = generate_llm_output_func(shape_types, shape_sizes)
        outputs.append(llm_output)

        dest_shape_info.append({'task_name': task_name,
                                'shape_types': shape_types,
                                'shape_sizes': shape_sizes})

    pd_dataset = pd.DataFrame({'input_data': inputs,
                               'user_prompt': user_prompts,
                               'output_data': outputs,
                               'dest_shape_info': dest_shape_info})

    return pd_dataset


# Deep Learning 모델 구조 관련 LLM 학습 데이터셋 생성
# Create Date : 2025.03.17
# Last Update Date : 2025.03.21
# - generate_dataset 함수에 task_name 인수 추가 반영

# Arguments:
# - dataset_size (int) : 데이터셋 규모

# Returns:
# - dl_dataset (Pandas DataFrame) : Deep Learning 모델 구조 관련 학습 데이터셋

def generate_dl_model_dataset(dataset_size):
    return generate_dataset(task_name='dl_model',
                            dataset_size=dataset_size,
                            generate_structure_func=generate_dl_model_structure,
                            generate_prompt_func=generate_dl_model_prompt,
                            generate_llm_output_func=generate_dl_model_llm_output)


# Flow Chart 구조 관련 LLM 학습 데이터셋 생성
# Create Date : 2025.03.18
# Last Update Date : 2025.03.21
# - generate_dataset 함수에 task_name 인수 추가 반영

# Arguments:
# - dataset_size (int) : 데이터셋 규모

# Returns:
# - flow_chart_dataset (Pandas DataFrame) : Flow Chart 구조 관련 학습 데이터셋
def generate_flow_chart_dataset(dataset_size):
    return generate_dataset(task_name='flowchart',
                            dataset_size=dataset_size,
                            generate_structure_func=generate_flow_chart_structure,
                            generate_prompt_func=generate_flow_chart_prompt,
                            generate_llm_output_func=generate_flow_chart_llm_output)
