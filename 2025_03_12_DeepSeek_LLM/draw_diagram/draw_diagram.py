import cv2
import numpy as np

WIDTH = 1000
HEIGHT = 600

canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255


# Diagram 의 도형 생성
# Create Date : 2025.03.15
# Last Update Date : -

# Arguments:
# - x      (int)   : 도형의 x 좌표
# - y      (int)   : 도형의 y 좌표
# - width  (int)   : 도형의 가로 길이
# - height (int)   : 도형의 세로 길이
# - shape  (str)   : 도형의 모양 (원, 직사각형 등)
# - color  (tuple) : 도형의 배경 색

# Returns:
# - canvas 에 해당 도형 추가

def generate_node(x, y, width, height, shape, color):
    global canvas


# Diagram 의 화살표 생성
# Create Date : 2025.03.15
# Last Update Date : -

# Arguments:
# - x0    (int)   : 시작점 도형의 x 좌표
# - y0    (int)   : 시작점 도형의 y 좌표
# - x1    (int)   : 끝점 도형의 x 좌표
# - y1    (int)   : 끝점 도형의 y 좌표
# - shape (str)   : 화살표의 모양 (실선, 점선 등)
# - color (tuple) : 화살표의 색

# Returns:
# - canvas 에 해당 화살표 추가

def generate_arrow(x0, y0, x1, y1, shape, color):
    global canvas
    raise NotImplementedError


# 각 line 을 읽고, 해당 line 의 정보를 이용하여 Diagram 에 도형 및 화살표 추가
# Create Date : 2025.03.15
# Last Update Date : -

# Arguments:
# - line_text (str) : 각 line 의 text 내용

# Returns:
# - canvas 에 해당 line 과 관련된 도형 추가

def generate_diagram_each_line(line_text):
    global canvas
    raise NotImplementedError


# 파일을 읽어서 해당 파일에 쓰인 각 line 을 파싱하여 도형 및 화살표 추가
# Create Date : 2025.03.15
# Last Update Date : -

# Arguments:
# - file_path (str) : 다이어그램 정보가 텍스트 형태로 저장된 파일 경로

# Returns:
# - canvas 에 해당 파일의 정보를 이용하여 도형 추가

def generate_diagram(file_path):
    global canvas
    raise NotImplementedError
