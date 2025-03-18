import draw_diagram as dd
import os

if __name__ == '__main__':
    diagram_txt_path = f'{os.path.abspath(os.path.dirname(__file__))}/diagram.txt'
    dd.generate_diagram(diagram_txt_path)
