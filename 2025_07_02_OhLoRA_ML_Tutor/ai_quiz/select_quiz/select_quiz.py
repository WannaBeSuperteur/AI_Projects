
import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
QUIZ_LIST_PATH = f'{PROJECT_DIR_PATH}/dataset/question_list.csv'


# 다음에 출제할 퀴즈 선택
# Create Date : 2025.07.21
# Last Update Date : -

# Arguments:
# - log_csv_path (str) : 퀴즈 풀이 로그 파일 경로

# Returns:
# - selected_quiz_no (int) : 다음에 선택할 퀴즈 번호 (dataset/question_list.csv 인덱스 기준)

def select_next_quiz(log_csv_path):
    log_csv = pd.read_csv(log_csv_path)
    print(log_csv)

    raise NotImplementedError

