import pandas as pd
import numpy as np
import random

MAX_PROMPT_SEED = 9999999
MAX_MODEL_STRUCTURE_SEED = 9999999
BACKGROUND_COLOR_LIST = [(255, 255, 255), (240, 240, 240), (224, 224, 224), (208, 208, 208),
                         (224, 224, 240), (192, 192, 224), (240, 232, 224), (224, 208, 192),
                         (224, 240, 232), (192, 224, 208), (232, 224, 240), (208, 192, 224)]
LINE_COLOR_LIST = [(0, 0, 0), (32, 32, 32), (64, 64, 64), (96, 96, 96),
                   (32, 32, 64), (64, 64, 128), (64, 48, 32), (128, 96, 64),
                   (32, 64, 48), (64, 128, 96), (48, 32, 64), (96, 64, 128)]

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 600

# for prompt engineering
PROMPT_PREFIX = "Represent below as a Python list.\n\n"
PROMPT_SUFFIX = """in the following format.

At this time, each node is represented in the format of Python list "[node No.,
X position (px), Y position (px), shape (rectangle, round rectangle or circle),
width (px), height (px), connection line shape (solid or dashed), background color,
connection line color, list of node No. s of other nodes pointed to by the connection line]".

At this time, the color is represented in the format of tuple (R, G, B), between 0 and 255, and
X position range is 0-1000 and Y position range is 0-600.

It is important to draw a representation of high readability."""


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


# Deep Learning 모델 구조 관련 사용자 입력 프롬프트 생성
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - prompt_seed (int)       : 프롬프트 종류를 나타내는 int 값 (0 - 9,999,999)
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

    user_prompt_part1_candidates = ['with', 'of', 'consist of']

    input_node_names = ['input layer nodes', 'input nodes', 'input elements', 'input size']
    hidden_layer_names = ['hidden layers', 'hiddens', 'intermediate layers', 'hidden layer', 'mid layers']
    output_node_names = ['output layer nodes', 'output nodes', 'output elements', 'output size']

    user_prompt = (user_prompt_part0_candidates[prompt_seed % 6] + ' ' +
                   user_prompt_part1_candidates[(prompt_seed // 6) % 3] + ' ')

    # 한 group (1~2 Conv. + 1 Pool.) 내의 모든 Conv. 레이어를 묶어서 한번에 표현
    say_conv_pool_at_once = random.random() < 0.4
    last_pooling_layer_idx = 0

    for idx, (t, s) in enumerate(zip(layer_types, layer_sizes)):
        additional_and = "and " if random.random() < 0.5 else ""
        additional_then = "then " if random.random() < 0.5 else ""
        additional_and_then = additional_and + additional_then

        # Conv. or Pool. layer
        if t in ['conv', 'pool']:
            conv = "convolutional" if random.random() < 0.5 else "conv"

            if random.random() < 0.4:
                pooling_type = "max " if random.random() < 0.5 else "average "
            else:
                pooling_type = ""

            if say_conv_pool_at_once:

                # Conv. + Pool. 그룹의 마지막 레이어인 Pooling Layer 에서만 해당 그룹의 종류 별 레이어 개수 체크
                if t == 'conv':
                    continue

                current_conv_and_pool_group = layer_types[last_pooling_layer_idx : idx+1]
                conv_layers = current_conv_and_pool_group.count('conv')
                last_pooling_layer_idx = idx

                if conv_layers > 1:
                    user_prompt += f'{conv_layers} 3 x 3 {conv} layers and a 2 x 2 {pooling_type}pooling layer, {additional_then}'
                else:
                    user_prompt += f'a 3 x 3 {conv} layer and a 2 x 2 {pooling_type}pooling layer, {additional_then}'

            else:
                r = random.random()

                # Convolutional Layer
                if t == 'conv':
                    if r < 0.2:
                        user_prompt += f'{conv} layer, {additional_and_then}'
                    elif r < 0.35:
                        user_prompt += f'3 x 3 {conv} layer, {additional_and_then}'
                    elif r < 0.5:
                        user_prompt += f'3x3 {conv} layer, {additional_and_then}'
                    elif r < 0.65:
                        user_prompt += f'3 * 3 {conv} layer, {additional_and_then}'
                    elif r < 0.8:
                        user_prompt += f'3*3 {conv} layer, {additional_and_then}'
                    elif r < 0.9:
                        user_prompt += f'{conv} layer (output is {s} x {s}), {additional_and_then}'
                    else:
                        user_prompt += f'{conv} layer (output: {s} x {s} feature map), {additional_and_then}'

                # Pooling Layer
                elif t == 'pool':
                    if r < 0.3:
                        user_prompt += f'{pooling_type}pooling layer, {additional_then}'
                    elif r < 0.475:
                        user_prompt += f'2 x 2 {pooling_type}pooling layer, {additional_then}'
                    elif r < 0.65:
                        user_prompt += f'2x2 {pooling_type}pooling layer, {additional_then}'
                    elif r < 0.825:
                        user_prompt += f'2 * 2 {pooling_type}pooling layer, {additional_then}'
                    else:
                        user_prompt += f'2*2 {pooling_type}pooling layer, {additional_then}'

        # CNN input layer
        elif t == 'cnn_input':
            additional_img = ' image' if random.random() < 0.5 else ''
            img_size = f'{s} * {s}' if random.random() < 0.5 else f'{s} x {s}'
            user_prompt += f'{img_size} input{additional_img}, '

        # input layer (dense)
        elif idx == 0:
            input_node_name = input_node_names[(prompt_seed // (6 * 3)) % 4]
            user_prompt += f'{s} {input_node_name}, '

        # output layer (dense)
        elif idx == len(layer_types) - 1:
            output_node_name = output_node_names[(prompt_seed // (6 * 3 * 4 * 5)) % 4]
            user_prompt += f'and {s} {output_node_name} '

        # last hidden layer (dense)
        elif idx == len(layer_types) - 2:
            hidden_layer_name = hidden_layer_names[(prompt_seed // (6 * 3 * 4)) % 5]

            if idx == 1 or layer_types[idx - 1] in ['conv', 'pool']:  # only one hidden layer
                user_prompt += f'and {s} nodes in {hidden_layer_name}, '

            else:
                r = random.random()
                additional_and = "and " if random.random() < 0.5 else ""
                hidden_layer_cnt = (len(layer_types) - 2) - (layer_types.count('conv') + layer_types.count('pool'))

                if r < 0.25:
                    user_prompt += f'{additional_and}{s} nodes in each of the {hidden_layer_cnt} {hidden_layer_name}, '
                elif r < 0.5:
                    user_prompt += f'{additional_and}{s} nodes in the {hidden_layer_name}, '
                elif r < 0.75:
                    user_prompt += f'{additional_and}{s} nodes in {hidden_layer_cnt} {hidden_layer_name}, '
                else:
                    user_prompt += f'{additional_and}{s} nodes in {hidden_layer_name}, '

        # hidden layer (dense)
        else:
            if random.random() < 0.5:
                user_prompt += f'{s}, '
            else:
                user_prompt += f'{s} '

    entire_prompt = PROMPT_PREFIX + user_prompt + PROMPT_SUFFIX
    return entire_prompt, user_prompt


# Deep Learning 모델 구조 관련 레이어 별 출력값을 생성하기 위한 shape 정보 계산
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - layer_size        (int)  : 해당 레이어의 크기 (node 개수 or feature map 크기)
# - layer_idx         (int)  : 해당 레이어의 인덱스 (모든 레이어 중)
# - layer_cnt         (int)  : 모든 레이어의 개수
# - max_layer_size    (int)  : 모든 레이어의 크기의 최댓값
# - diagram_direction (str)  : 다이어그램 그리기 방향 (가로, 세로)
# - draw_each_node    (bool) : 각 레이어의 node (Conv., Pool. 제외) 를 표시할 것인가?

# Returns:
# - shapes_info (dict) : shape 정보,
#                        {'x': x (center), 'y': y (center), 'w': width, 'h': height}

def generate_shapes_info_of_layer(layer_size, layer_idx, layer_cnt, diagram_direction, draw_each_node, max_layer_size):

    shapes_info = []
    layer_lengths_relative = np.log(layer_size + 1) / np.log(max_layer_size + 1)

    if diagram_direction == 'horizontal':
        layer_x = (layer_idx + 1) / CANVAS_WIDTH * (layer_cnt + 1)
        layer_y = CANVAS_HEIGHT // 2
        layer_width = CANVAS_WIDTH // (2 * layer_cnt)
        layer_height = int(0.6 * CANVAS_HEIGHT * layer_lengths_relative)

        if draw_each_node:
            node_xs = [layer_x for _ in range(layer_size)]
            node_ys = np.linspace(CANVAS_HEIGHT // 2 - layer_height // 2,
                                  CANVAS_HEIGHT // 2 + layer_height // 2,
                                  layer_size)[1:-1]
            node_widths = [layer_width for _ in range(layer_size)]
            node_heights = [layer_height for _ in range(layer_size)]

            for x, y, w, h in zip(node_xs, node_ys, node_widths, node_heights):
                shapes_info.append({'x': x, 'y': y, 'w': w, 'h': h})

    else:  # vertical
        layer_x = CANVAS_WIDTH // 2
        layer_y = (layer_idx + 1) / CANVAS_HEIGHT * (layer_cnt + 1)
        layer_width = int(0.6 * CANVAS_WIDTH * layer_lengths_relative)
        layer_height = CANVAS_HEIGHT // (2 * layer_cnt)

        if draw_each_node:
            node_xs = np.linspace(CANVAS_WIDTH // 2 - layer_width // 2,
                                  CANVAS_WIDTH // 2 + layer_width // 2,
                                  layer_size)[1:-1]
            node_ys = [layer_y for _ in range(layer_size)]
            node_widths = [layer_width for _ in range(layer_size)]
            node_heights = [layer_height for _ in range(layer_size)]

            for x, y, w, h in zip(node_xs, node_ys, node_widths, node_heights):
                shapes_info.append({'x': x, 'y': y, 'w': w, 'h': h})

    if not draw_each_node:
        shapes_info.append({'x': layer_x,
                            'y': layer_y,
                            'w': layer_width,
                            'h': layer_height})

    return shapes_info


# Deep Learning 모델 구조 관련 모델 출력값 생성
# Create Date : 2025.03.17
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

    property_each_layer_type = {'cnn_input': {'back': back_colors[0], 'line': line_colors[0], 'shape': shape[0]},
                                'conv': {'back': back_colors[1], 'line': line_colors[1], 'shape': shape[1]},
                                'pool': {'back': back_colors[2], 'line': line_colors[2], 'shape': shape[2]},
                                'dense_input': {'back': back_colors[3], 'line': line_colors[3], 'shape': shape[3]},
                                'dense_hidden': {'back': back_colors[4], 'line': line_colors[4], 'shape': shape[4]},
                                'dense_output': {'back': back_colors[5], 'line': line_colors[5], 'shape': shape[5]}}

    # generate shape info for each layer
    for layer_idx, (layer_type, layer_size) in enumerate(zip(layer_types, layer_sizes)):
        shapes_info = generate_shapes_info_of_layer(layer_size=layer_size,
                                                    layer_idx=layer_idx,
                                                    layer_cnt=n,
                                                    diagram_direction=diagram_direction,
                                                    draw_each_node=draw_each_node,
                                                    max_layer_size=max(layer_sizes))
        shapes_info_dict[layer_idx] = shapes_info

    # generate output for each layer
    for layer_idx, (layer_type, layer_size) in enumerate(zip(layer_types, layer_sizes)):

        # TODO: implement

        model_output += model_output_of_layer + '\n'

    return model_output


# Deep Learning 모델 구조 관련 LLM 학습 데이터셋 생성
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - dataset_size (int) : 데이터셋 규모

# Returns:
# - dl_dataset (Pandas DataFrame) : Deep Learning 모델 구조 관련 학습 데이터셋

def generate_dl_model_dataset(dataset_size):
    inputs = []
    outputs = []
    user_prompts = []

    # 데이터셋 생성
    for i in range(dataset_size):

        # generate user prompt (LLM input)
        layer_types, layer_sizes = generate_dl_model_structure(random.randint(0, MAX_MODEL_STRUCTURE_SEED))
        entire_prompt, user_prompt = generate_dl_model_prompt(random.randint(0, MAX_PROMPT_SEED), layer_types, layer_sizes)
        inputs.append(entire_prompt)
        user_prompts.append(user_prompt)

        # generate LLM output for training
        llm_output = generate_dl_model_llm_output(layer_types, layer_sizes)
        outputs.append(llm_output)

    dl_dataset = pd.DataFrame({'input_data': inputs, 'user_prompt': user_prompts, 'output_data': outputs})
    return dl_dataset
