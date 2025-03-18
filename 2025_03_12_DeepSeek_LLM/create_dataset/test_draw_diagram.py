import cv2
import pandas as pd

import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from draw_diagram.draw_diagram import generate_diagram_from_lines


# csv 파일에 있는 Diagram 을 읽어서 test_diagrams 디렉토리에 다이어그램 저장
# Create Date : 2025.03.18
# Last Update Date : -

def generate_diagram_from_csv():
    abs_path = os.path.abspath(os.path.dirname(__file__))

    df = pd.read_csv(f'{abs_path}/sft_dataset.csv')
    outputs = df['output_data'].tolist()
    os.makedirs(f'{abs_path}/test_diagrams', exist_ok=True)

    for idx, output in enumerate(outputs):
        lines = str(output).split('\n')
        output_path = f'{abs_path}/test_diagrams/diagram_{idx:06d}.png'
        generate_diagram_from_lines(lines, output_path)


if __name__ == '__main__':
    generate_diagram_from_csv()
