
try:
    from select_quiz.select_quiz import select_next_quiz
except:
    from ai_quiz.select_quiz.select_quiz import select_next_quiz


import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
QUIZ_LIST_PATH = f'{PROJECT_DIR_PATH}/ai_quiz/dataset/question_list.csv'


if __name__ == '__main__':
    quiz_log_csv_path = f'{PROJECT_DIR_PATH}/ai_quiz/select_quiz/example_log.csv'
    next_quiz_no = select_next_quiz(quiz_log_csv_path)

    quiz_list = pd.read_csv(QUIZ_LIST_PATH)
    next_quiz = quiz_list['quiz'][next_quiz_no]

    print(f'next quiz : {next_quiz} (idx: {next_quiz_no})')
