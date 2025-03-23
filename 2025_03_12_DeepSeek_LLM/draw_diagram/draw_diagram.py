import cv2
import numpy as np
import math

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common_values import CANVAS_WIDTH as WIDTH, CANVAS_HEIGHT as HEIGHT

try:
    from diagram_format_finder import find_diagram_formats, add_diagram_info
except:
    from draw_diagram.diagram_format_finder import find_diagram_formats, add_diagram_info

DASH_INTERVAL = 20  # interval for dashed line
LINE_MARGIN = 4

canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
diagram_dict = {}


# Round Rectangle 그리기
# Create Date : 2025.03.16
# Last Update Date : 2025.03.18
# - global canvas 누락 해결

# Arguments:
# - x         (int)   : 도형의 x 좌표
# - y         (int)   : 도형의 y 좌표
# - width     (int)   : 도형의 가로 길이
# - height    (int)   : 도형의 세로 길이
# - color     (tuple) : 도형의 배경 색 (B-G-R 순)
# - thickness (int)   : 도형의 굵기 (-1 이면 배경색 칠함)

# Returns:
# - canvas 에 해당 Round Rectangle 추가

def draw_round_rectangle(x, y, width, height, color, thickness):
    global canvas

    round_radius = int(min(0.1 * max(width, height), 0.5 * min(width, height)))

    top = y - int(height / 2)
    left = x - int(width / 2)
    bottom = y + int(height / 2)
    right = x + int(width / 2)

    # draw circles
    circle_centers = [
        (left + round_radius, top + round_radius),     # top-left circle
        (right - round_radius, top + round_radius),    # top-right circle
        (left + round_radius, bottom - round_radius),  # bottom-left circle
        (right - round_radius, bottom - round_radius)  # bottom-right circle
    ]

    for circle_center in circle_centers:
        cv2.circle(canvas,
                   center=circle_center,
                   radius=round_radius,
                   color=color,
                   thickness=thickness,
                   lineType=cv2.LINE_AA)

    # draw rectangles
    cv2.rectangle(canvas,
                  pt1=(left + round_radius, top),
                  pt2=(right - round_radius, bottom),
                  color=color,
                  thickness=thickness)

    cv2.rectangle(canvas,
                  pt1=(left, top + round_radius),
                  pt2=(right, bottom - round_radius),
                  color=color,
                  thickness=thickness)


# Diagram 의 도형 생성
# Create Date : 2025.03.16
# Last Update Date : 2025.03.17
# - circle 그릴 때 anti-aliasing 누락 수정 (lineType=cv2.LINE_AA)
# - rectangle 그릴 때 pt1, pt2 의 좌표를 int 자료형이 되도록 수정
# - 기타 버그 수정 (도형 그리기 순서, ellipse size 등)

# Arguments:
# - x      (int)   : 도형의 x 좌표
# - y      (int)   : 도형의 y 좌표
# - width  (int)   : 도형의 가로 길이
# - height (int)   : 도형의 세로 길이
# - shape  (str)   : 도형의 모양 (원, 직사각형 등)
# - color  (tuple) : 도형의 배경 색 (B-G-R 순)

# Returns:
# - canvas 에 해당 도형 추가

def generate_node(x, y, width, height, shape, color):
    global canvas

    colors = [(0, 0, 0), color]  # edge / background line color
    thickness = [3, -1]          # thickness (3 for edge line / -1 for background)

    for c, t in zip(colors, thickness):

        # round rectangle
        if 'round' in shape and 'rect' in shape:
            draw_round_rectangle(x, y, width, height, color=c, thickness=t)

        # rectangle
        elif 'rect' in shape:
            cv2.rectangle(canvas,
                          pt1=(x - int(width / 2), y - int(height / 2)),
                          pt2=(x + int(width / 2), y + int(height / 2)),
                          color=c,
                          thickness=t)

        # circle
        elif 'circle' in shape:
            cv2.ellipse(canvas,
                        center=(x, y),
                        axes=(int(width / 2), int(height / 2)),
                        angle=0,
                        startAngle=0,
                        endAngle=360,
                        color=c,
                        thickness=t,
                        lineType=cv2.LINE_AA)


# Diagram 의 화살표 생성 시, 화살표의 끝점의 좌표 계산
# Create Date : 2025.03.16
# Last Update Date : 2025.03.17
# - destination point 계산 오류 및 margin 수정

# Arguments:
# - x0          (int)   : 시작점 도형의 x 좌표
# - y0          (int)   : 시작점 도형의 y 좌표
# - x1          (int)   : 끝점 도형의 x 좌표
# - y1          (int)   : 끝점 도형의 y 좌표
# - dest_shape  (int)   : 끝점 도형의 모양
# - dest_width  (int)   : 끝점 도형의 가로 길이
# - dest_height (int)   : 끝점 도형의 세로 길이

# Returns:
# - x_dest (float) : 화살표 끝점의 x 좌표
# - y_dest (float) : 화살표 끝점의 y 좌표

def compute_dest_point(x0, y0, x1, y1, dest_shape, dest_width, dest_height):
    x_dest, y_dest = None, None

    width_shift = dest_width / 2 + LINE_MARGIN
    height_shift = dest_height / 2 + LINE_MARGIN

    # round rectangle or rectangle
    if 'rect' in dest_shape:
        if abs(x0 - x1) >= abs(y0 - y1):
            x_dest = x1 - width_shift if x0 < x1 else x1 + width_shift
            y_dest = y1
        else:
            x_dest = x1
            y_dest = y1 - height_shift if y0 < y1 else y1 + height_shift

    # circle
    elif 'circle' in dest_shape:
        if y0 == y1 and x0 == x1:
            x_dest = x1
            y_dest = y1

        elif x0 == x1:
            x_dest = x1
            y_dest = y1 - height_shift if y0 < y1 else y1 + height_shift

        elif y0 == y1:
            x_dest = x1 - width_shift if x0 < x1 else x1 + width_shift
            y_dest = y1

        else:
            angle = math.atan((y1 - y0) / (x1 - x0))
            abs_cos = abs(math.cos(angle))
            abs_sin = abs(math.sin(angle))

            x_dest = x1 - abs_cos * width_shift if x0 < x1 else x1 + abs_cos * width_shift
            y_dest = y1 - abs_sin * height_shift if y0 < y1 else y1 + abs_sin * height_shift

    return x_dest, y_dest


# Diagram 의 점선 화살표를 그리기 위한 dash dot 의 가로, 세로 성분 길이 구하기
# Create Date : 2025.03.16
# Last Update Date : 2025.03.17
# - dash width, height 계산 버그 수정

# Arguments:
# - x0     (int)   : 시작점 도형의 x 좌표
# - y0     (int)   : 시작점 도형의 y 좌표
# - x_dest (float) : 화살표 끝점의 x 좌표
# - y_dest (float) : 화살표 끝점의 y 좌표

# Returns:
# - dash_width  (float) : dash dot 의 가로 성분 길이
# - dash_height (float) : dash dot 의 세로 성분 길이

def compute_dash_width_and_height(x0, y0, x_dest, y_dest):
    dash_length = DASH_INTERVAL / 2.0

    if x0 == x_dest:
        dash_width = 0
        dash_height = dash_length if y0 < y_dest else (-1) * dash_length

    elif y0 == y_dest:
        dash_width = dash_length if x0 < x_dest else (-1) * dash_length
        dash_height = 0

    else:
        angle = math.atan2(y_dest - y0, x_dest - x0)
        dash_width = math.cos(angle) * dash_length
        dash_height = math.sin(angle) * dash_length

    return dash_width, dash_height


# Diagram 의 점선 그리기
# Create Date : 2025.03.16
# Last Update Date : 2025.03.18
# - 3월 17일 신규 요구사항 (연결선의 종류 변경) 반영
# - global canvas 누락 해결

# Arguments:
# - x0         (int)   : 시작점 도형의 x 좌표
# - y0         (int)   : 시작점 도형의 y 좌표
# - x_dest     (float) : 점선 끝점의 x 좌표
# - y_dest     (float) : 점선 끝점의 y 좌표
# - line_color (tuple) : 점선의 색 (B-G-R 순)

# Returns:
# - canvas 에 해당 점선 추가

def generate_dashed_line(x0, y0, x_dest, y_dest, line_color):
    global canvas

    if y0 == y_dest and x0 == x_dest:
        return

    # compute width and height of dash
    dash_width, dash_height = compute_dash_width_and_height(x0, y0, x_dest, y_dest)

    # draw dash
    x = x0
    y = y0

    while True:
        cv2.line(canvas,
                 pt1=(int(x), int(y)),
                 pt2=(int(x + dash_width), int(y + dash_height)),
                 color=line_color,
                 thickness=1,
                 lineType=cv2.LINE_AA)

        x_next = x + 2.0 * dash_width
        y_next = y + 2.0 * dash_height

        # when passed destination point
        if (x - x_dest) * (x_next - x_dest) <= 0 and (y - y_dest) * (y_next - y_dest) <= 0:
            return

        x = x_next
        y = y_next


# Diagram 의 연결선 생성
# Create Date : 2025.03.16
# Last Update Date : 2025.03.18
# - 3월 17일 신규 요구사항 (연결선의 종류 변경) 반영
# - tip length 수정

# Arguments:
# - x0          (int)   : 시작점 도형의 x 좌표
# - y0          (int)   : 시작점 도형의 y 좌표
# - x1          (int)   : 끝점 도형의 x 좌표
# - y1          (int)   : 끝점 도형의 y 좌표
# - line_shape  (str)   : 연결선의 모양 (solid arrow, solid line, dashed line)
# - line_color  (tuple) : 연결선의 색 (B-G-R 순)
# - dest_shape  (int)   : 끝점 도형의 모양
# - dest_width  (int)   : 끝점 도형의 가로 길이
# - dest_height (int)   : 끝점 도형의 세로 길이

# Returns:
# - canvas 에 해당 화살표 추가

def generate_line(x0, y0, x1, y1, line_shape, line_color, dest_shape, dest_width, dest_height):
    global canvas

    x_dest, y_dest = compute_dest_point(x0, y0, x1, y1, dest_shape, dest_width, dest_height)

    # solid arrow
    if 'solid' in line_shape and 'arrow' in line_shape:
        distance = math.sqrt((x_dest - x0) ** 2.0 + (y_dest - y0) ** 2.0)

        cv2.arrowedLine(canvas,
                        pt1=(x0, y0),
                        pt2=(int(x_dest), int(y_dest)),
                        color=line_color,
                        thickness=1,
                        line_type=cv2.LINE_AA,
                        tipLength=max(10.0 / distance, 0.05))

    # solid line
    elif 'solid' in line_shape and 'line' in line_shape:
        cv2.line(canvas,
                 pt1=(x0, y0),
                 pt2=(int(x_dest), int(y_dest)),
                 color=line_color,
                 thickness=1,
                 lineType=cv2.LINE_AA)

    # dotted (dashed) line
    elif 'dash' in line_shape or 'dot' in line_shape:
        generate_dashed_line(x0, y0, x_dest, y_dest, line_color)


# 각 line 을 읽고, 해당 line 의 정보를 이용하여 Diagram 에 도형 및 화살표 추가
# Create Date : 2025.03.17
# Last Update Date : 2025.03.21
# - add_diagram_info 함수 변경으로 diagram_dict 를 인수로 추가

# Arguments:
# - line_text (str) : 각 line 의 text 내용

# Returns:
# - canvas 에 해당 line 과 관련된 도형 추가

# 참고:
# - LLM 보안 (프롬프트를 이용한 해킹) 등의 이슈 우려로 인해 Python 의 eval() 함수를 사용하지 않음

def generate_diagram_each_line(line_text):
    global diagram_dict

    # remove quotes
    line_text = line_text.replace('"', '').replace("'", "")

    # find diagram shape line format from line text
    diagram_formats = find_diagram_formats(line_text)
    add_diagram_info(diagram_formats, diagram_dict)

    # draw lines first
    for node_no, info in diagram_dict.items():
        for connected_node_no in info['connected_nodes']:

            # source = dest 인 경우 pass
            if node_no == connected_node_no:
                continue

            if connected_node_no in diagram_dict.keys():
                dest_node_info = diagram_dict[connected_node_no]

                generate_line(x0=info['shape_x'],
                              y0=info['shape_y'],
                              x1=dest_node_info['shape_x'],
                              y1=dest_node_info['shape_y'],
                              line_shape=info['line_shape'],
                              line_color=info['line_color'],
                              dest_shape=dest_node_info['shape'],
                              dest_width=dest_node_info['shape_width'],
                              dest_height=dest_node_info['shape_height'])

    # draw diagram
    for node_no, info in diagram_dict.items():
        generate_node(x=info['shape_x'],
                      y=info['shape_y'],
                      width=info['shape_width'],
                      height=info['shape_height'],
                      shape=info['shape'],
                      color=info['shape_color'])


# 읽어온 파일의 line 들을 각각의 line 으로 파싱하여 도형 및 화살표 추가
# Create Date : 2025.03.18
# Last Update Date : 2025.03.19
# - output_path -> save_path 로 변수명 통일

# Arguments:
# - lines     (list(str)) : 다이어그램 정보가 텍스트 형태로 저장된 파일 경로
# - save_path (str)       : 다이어그램 파일 저장 경로

# Returns:
# - canvas 에 해당 파일의 정보를 이용하여 도형 추가
# - 해당 canvas 를 이미지 파일로 저장

def generate_diagram_from_lines(lines, save_path):
    global canvas, diagram_dict

    # 캔버스 및 diagram dict 초기화
    canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    diagram_dict = {}

    for line_idx, line_text in enumerate(lines):
        try:
            generate_diagram_each_line(line_text)
        except Exception as e:
            print(f'line {line_idx} : {e}')

    # 파일 저장
    cv2.imwrite(save_path, canvas)


# 파일을 읽어서 해당 파일에 쓰인 각 line 을 파싱하여 도형 및 화살표 추가
# Create Date : 2025.03.16
# Last Update Date : 2025.03.19
# - output 경로 수정
# - merge conflict 해결

# Arguments:
# - file_path (str) : 다이어그램 정보가 텍스트 형태로 저장된 파일 경로
# - save_path (str) : 다이어그램 이미지 저장 경로

# Returns:
# - canvas 에 해당 파일의 정보를 이용하여 도형 추가
# - 해당 canvas 를 이미지 파일로 저장

def generate_diagram(file_path, save_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    generate_diagram_from_lines(lines, save_path)


