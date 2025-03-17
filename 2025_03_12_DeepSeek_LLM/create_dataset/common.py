import pandas as pd
import random

MAX_PROMPT_SEED = 9999999
MAX_MODEL_STRUCTURE_SEED = 9999999


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
# - layer_type (list(str)) : 딥러닝 모델의 각 레이어의 종류
# - layer_size (list(int)) : 딥러닝 모델의 각 레이어의 크기 (node 개수 or feature map 크기)

def generate_model_structure(layer_config_seed):

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
            hidden_nodes = [(dense_config_seed // (5 * 3)) % 6 + 1,
                            (dense_config_seed // (5 * 3 * 6)) % 6 + 1]

        else:  # hidden_layers == 3
            hidden_nodes = [(dense_config_seed // (5 * 3)) % 6 + 1,
                            (dense_config_seed // (5 * 3 * 6)) % 10 + 1,
                            (dense_config_seed // (5 * 3 * 6 * 10)) % 6 + 1]

        # output layer
        output_nodes = (layer_config_seed // (5 * 3 * 6 * 10 * 6)) % 2 + 1

        layer_type = ['dense'] * (hidden_layers + 2)
        layer_size = [input_nodes] + hidden_nodes + [output_nodes]

    # Conv. + Pool. + Dense (Convolutional Neural Network)
    else:
        conv_config_seed = layer_config_seed // 3

        # input layer
        input_sizes = [28, 32, 64, 128, 224, 256, 512, 768]
        conv_pool_times = [3, 3, 4, 5, 6, 6, 7, 8]

        input_size = input_sizes[conv_config_seed % 8]
        conv_pool_cnt = conv_pool_times[conv_config_seed % 8]

        # size after Conv. and Pool. layers
        conv_pool_size = []
        current_size = input_size

        for c in range(conv_pool_cnt):
            current_size -= 2
            conv_pool_size.append(current_size)
            current_size = current_size // 2
            conv_pool_size.append(current_size)

        # hidden layer (dense after Conv. + Pool, fully-connected)
        hidden_layers = (conv_config_seed // 8) % 3 + 1

        if hidden_layers == 1:
            hidden_node_counts = [[64], [128], [256]]
            hidden_nodes = hidden_node_counts[(conv_config_seed // (8 * 3)) % 3]

        elif hidden_layers == 2:
            hidden_node_counts = [[256, 32], [512, 64], [512, 128], [1024, 256]]
            hidden_nodes = hidden_node_counts[(conv_config_seed // (8 * 3)) % 4]

        else:  # hidden_layers == 3
            hidden_node_counts = [[256, 64, 16], [512, 128, 16], [512, 256, 64], [1024, 256, 64]]
            hidden_nodes = hidden_node_counts[(conv_config_seed // (8 * 3)) % 4]

        # output layer
        output_nodes = (layer_config_seed // (8 * 3 * 3 * 4)) % 2 + 1

        layer_type = [input_size] + ['conv', 'pool'] * conv_pool_cnt + ['dense'] * (hidden_layers + 1)
        layer_size = [input_size] + conv_pool_size + hidden_nodes + [output_nodes]

    return layer_type, layer_size


# Deep Learning 모델 구조 관련 사용자 입력 프롬프트 생성
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - prompt_seed (int)       : 프롬프트 종류를 나타내는 int 값 (0 - 9,999,999)
# - layer_type  (list(str)) : 딥러닝 모델의 각 레이어의 종류
# - layer_size  (list(int)) : 딥러닝 모델의 각 레이어의 크기 (node 개수 or feature map 크기)

# Returns:
# - entire_prompt (str) : Deep Learning 모델 구조 관련 학습 데이터셋의 입력 프롬프트
# - user_prompt   (str) : Prompt Engineering 을 위한 앞뒤 부분을 제외한 순수 유저 프롬프트

def generate_prompt(prompt_seed, layer_type, layer_size):
    user_prompt_part0_candidates = ['A deep learning model', 'DL model', 'neural network', 'NN']
    user_prompt_part1_candidates = ['with', 'of', 'consist of']

    input_node_names = ['input layer nodes', 'input nodes', 'input elements', 'input size']
    hidden_layer_names = ['hidden layers', 'hiddens', 'intermediate layers', 'hidden layer', 'mid layers']
    output_node_names = ['output layer nodes', 'output nodes', 'output elements', 'output size']

    user_prompt = (user_prompt_part0_candidates[prompt_seed % 4] + ' ' +
                   user_prompt_part1_candidates[(prompt_seed // 4) % 3] + ' ')

    # 모든 Conv. + Pool. 레이어를 묶어서 한번에 표현
    say_conv_pool_at_once = random.random() < 0.4
    said_conv_pool_once = False

    for idx, (t, s) in enumerate(zip(layer_type, layer_size)):
        additional_and = "and " if random.random() < 0.5 else ""
        additional_then = "then " if random.random() < 0.5 else ""
        additional_and_then = additional_and + additional_then

        # Conv. or Pool. layer
        if t in ['conv', 'pool']:
            conv = "convolutional" if random.random() < 0.5 else "conv"
            pooling_type = "max" if random.random() < 0.5 else "average"

            if say_conv_pool_at_once:
                if not said_conv_pool_once:
                    said_conv_pool_once = True
                    conv_layers = layer_type.count('conv')
                    pool_layers = layer_type.count('pool')

                    user_prompt += f'{conv_layers} 3 x 3 {conv} layers and {pool_layers} 2 x 2 {pooling_type} pooling layers, {additional_then}'

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
                        user_prompt += f'{pooling_type} pooling layer, {additional_then}'
                    elif r < 0.475:
                        user_prompt += f'2 x 2 {pooling_type} pooling layer, {additional_then}'
                    elif r < 0.65:
                        user_prompt += f'2x2 {pooling_type} pooling layer, {additional_then}'
                    elif r < 0.825:
                        user_prompt += f'2 * 2 {pooling_type} pooling layer, {additional_then}'
                    else:
                        user_prompt += f'2*2 {pooling_type} pooling layer, {additional_then}'

        # input layer (dense)
        elif idx == 0:
            user_prompt += f'{s} {input_node_names[(prompt_seed // (4 * 3)) % 4]}, '

        # output layer (dense)
        elif idx == len(layer_type) - 1:
            user_prompt += f'and {s} {output_node_names[(prompt_seed // (4 * 3 * 4 * 5)) % 4]} '

        # last hidden layer (dense)
        elif idx == len(layer_type) - 2:
            if idx == 1 or layer_type[idx - 1] in ['conv', 'pool']:  # only one hidden layer
                user_prompt += f'and {s} nodes in {hidden_layer_names[(prompt_seed // (4 * 3 * 4)) % 5]}, '
            else:
                r = random.random()
                additional_and = "and " if random.random() < 0.5 else ""
                hidden_layer_name = hidden_layer_names[(prompt_seed // (4 * 3 * 4)) % 5]
                hidden_layer_cnt = (len(layer_type) - 2) - (layer_type.count('conv') + layer_type.count('pool'))

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


# Deep Learning 모델 구조 관련 모델 출력값 생성
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - layer_type (list(str)) : 딥러닝 모델의 각 레이어의 종류
# - layer_size (list(int)) : 딥러닝 모델의 각 레이어의 크기 (node 개수 or feature map 크기)

# Returns:
# - model_output (str) : 다이어그램 형식의 텍스트 (draw_diagram/diagram.txt 참고)

def generate_llm_output(layer_type, layer_size):
    return ''  # temp


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
        layer_type, layer_size = generate_model_structure(random.randint(0, MAX_MODEL_STRUCTURE_SEED))
        entire_prompt, user_prompt = generate_prompt(random.randint(0, MAX_PROMPT_SEED), layer_type, layer_size)
        inputs.append(entire_prompt)
        user_prompts.append(user_prompt)

        # generate LLM output for training
        llm_output = generate_llm_output(layer_type, layer_size)
        outputs.append(llm_output)

    dl_dataset = pd.DataFrame({'input_data': inputs, 'user_prompt': user_prompts, 'output_data': outputs})
    return dl_dataset
