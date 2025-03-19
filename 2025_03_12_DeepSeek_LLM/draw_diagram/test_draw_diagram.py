import draw_diagram as dd
import os

if __name__ == '__main__':
    diagram_txt_path = f'{os.path.abspath(os.path.dirname(__file__))}/diagram.txt'
    diagram_save_path = f'{os.path.abspath(os.path.dirname(__file__))}/diagram.png'

    dd.generate_diagram(file_path=diagram_txt_path, save_path=diagram_save_path)
