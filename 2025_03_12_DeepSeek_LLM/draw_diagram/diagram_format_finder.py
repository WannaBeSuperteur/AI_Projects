# 각 line 에서 diagram format 의 텍스트 찾기
# Create Date : 2025.03.17
# Last Update Date : -

# Arguments:
# - line_text (str) : 각 line 의 text 내용

# Returns:
# - diagram_formats (list(str)) : diagram format 텍스트의 리스트

def find_diagram_formats(line_text):
    diagram_format_start_idx = 0
    diagram_formats = []
    braket_count = 0
    parentheses_count = 0

    for i in range(len(line_text)):
        if line_text[i] == '[':
            braket_count += 1
            if braket_count == 1:
                diagram_format_start_idx = i

        elif line_text[i] == ']':
            braket_count -= 1
            if braket_count == 0:
                diagram_format = line_text[diagram_format_start_idx: i + 1]
                diagram_formats.append(diagram_format)

        elif line_text[i] == '(':
            parentheses_count += 1

        elif line_text[i] == ')':
            parentheses_count -= 1

        # (R, G, B) 부분 및 [connected_node1, connected_node2, ...] 부분 처리
        elif line_text[i] == ',':
            if parentheses_count == 1:  # (R, G, B) 부분
                line_text = line_text[:i] + '@' + line_text[i+1:]

            if braket_count == 2:  # [connected_node1, connected_node2, ...] 부분
                line_text = line_text[:i] + '$' + line_text[i+1:]

    return diagram_formats


# 각 line 에서 찾은 diagram format 의 텍스트를 이용하여, 이를 이용하여 diagram dictionary 에 정보 추가
# Create Date : 2025.03.17
# Last Update Date : 2025.03.21
# - LLM Fine-tuning 결과에 대한 평가 시 사용하기 위해 diagram_dict 를 인수로 추가
# - shape 의 x, y, width, height 이 float 인 경우 처리

# Arguments:
# - diagram_formats (list(str)) : diagram format 텍스트의 리스트
# - diagram_dict    (dict)      : Diagram 도형 정보를 저장한 dictionary

# Returns:
# - diagram_dict 에 각각의 diagram format 텍스트의 내용으로부터 추출한 도형 정보를 추가

def add_diagram_info(diagram_formats, diagram_dict):
    for diagram_format in diagram_formats:

        # split by ","
        diagram_format_split = diagram_format.replace(', ', ',')[1:-1].split(',')
        node_no = diagram_format_split[0]

        shape_x = int(float(diagram_format_split[1]))
        shape_y = int(float(diagram_format_split[2]))
        shape = diagram_format_split[3]
        shape_width = int(float(diagram_format_split[4]))
        shape_height = int(float(diagram_format_split[5]))

        line_shape = diagram_format_split[6]

        # R-G-B -> B-G-R color format change
        shape_color_rgb = list(map(int, diagram_format_split[7][1:-1].replace('@ ', '@').split('@')))
        line_color_rgb = list(map(int, diagram_format_split[8][1:-1].replace('@ ', '@').split('@')))
        shape_color = (shape_color_rgb[2], shape_color_rgb[1], shape_color_rgb[0])
        line_color = (line_color_rgb[2], line_color_rgb[1], line_color_rgb[0])

        connected_nodes = diagram_format_split[9][1:-1].replace('$ ', '$').split('$')

        diagram_dict[node_no] = {
            'shape_x': shape_x,
            'shape_y': shape_y,
            'shape': shape,
            'shape_width': shape_width,
            'shape_height': shape_height,
            'line_shape': line_shape,
            'shape_color': shape_color,
            'line_color': line_color,
            'connected_nodes': connected_nodes
        }