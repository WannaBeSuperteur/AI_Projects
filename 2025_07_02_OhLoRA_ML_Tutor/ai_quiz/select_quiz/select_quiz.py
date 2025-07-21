
import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
QUIZ_LIST_PATH = f'{PROJECT_DIR_PATH}/ai_quiz/dataset/question_list.csv'


# 2개의 퀴즈에 대한 키워드를 비교하여 IoU 값 계산
# Create Date : 2025.07.21
# Last Update Date : -

# Arguments:
# - keywords_1 (str) : 1번째 퀴즈의 키워드 목록 ('|' 으로 구분)
# - keywords_2 (str) : 2번째 퀴즈의 키워드 목록 ('|' 으로 구분)

# Returns:
# - iou_score (float) : 해당 2개의 퀴즈에 대한 IoU Score 계산 결과

def compute_iou_score(keywords_1, keywords_2):
    keywords_1_list = keywords_1.split('|')
    keywords_2_list = keywords_2.split('|')
    keywords_1_set = set(keywords_1_list)
    keywords_2_set = set(keywords_2_list)

    union_size = len(keywords_1_set.union(keywords_2_set))
    intersection_size = len(keywords_1_set.intersection(keywords_2_set))

    iou_score = intersection_size / union_size
    return iou_score


# 다음에 출제할 퀴즈 선택
# Create Date : 2025.07.21
# Last Update Date : -

# Arguments:
# - log_csv_path (str) : 퀴즈 풀이 로그 파일 경로

# Returns:
# - selected_quiz_no (int) : 다음에 선택할 퀴즈 번호 (dataset/question_list.csv 인덱스 기준)

def select_next_quiz(log_csv_path):
    log_csv = pd.read_csv(log_csv_path)
    quiz_list_csv = pd.read_csv(QUIZ_LIST_PATH, index_col=0)

    logged_quiz_cnt = len(log_csv)
    quiz_cnt = len(quiz_list_csv)
    logged_quiz_mean_score = log_csv['score'].mean()

    print(log_csv)
    print(quiz_list_csv)

    # compute score for each quiz question
    iou = [0.0 for _ in range(quiz_cnt)]
    sum_score_weighted_by_iou = [0.0 for _ in range(quiz_cnt)]
    score_weighted_by_iou = [0.0 for _ in range(quiz_cnt)]

    for quiz_no in range(quiz_cnt):
        quiz_kwds = quiz_list_csv['keywords'][quiz_no]

        for logged_quiz_no in range(logged_quiz_cnt):
            logged_quiz_kwds = log_csv['keywords'][logged_quiz_no]
            logged_quiz_score = log_csv['score'][logged_quiz_no]
            iou_score = compute_iou_score(quiz_kwds, logged_quiz_kwds)

            iou[quiz_no] += iou_score
            sum_score_weighted_by_iou[quiz_no] += iou_score * logged_quiz_score

        if iou[quiz_no] > 0:
            score_weighted_by_iou[quiz_no] = sum_score_weighted_by_iou[quiz_no] / iou[quiz_no]
        else:
            score_weighted_by_iou[quiz_no] = logged_quiz_mean_score

    # aggregate compute result
    is_logged = [False for _ in range(quiz_cnt)]
    for quiz_no in log_csv['quiz_no']:
        is_logged[quiz_no] = True

    score_compute_result_dict = {'idx': list(range(quiz_cnt)),
                                 'logged': is_logged,
                                 'keywords': quiz_list_csv['keywords'],
                                 'score': score_weighted_by_iou}
    score_compute_result_df = pd.DataFrame(score_compute_result_dict)

    raise NotImplementedError

