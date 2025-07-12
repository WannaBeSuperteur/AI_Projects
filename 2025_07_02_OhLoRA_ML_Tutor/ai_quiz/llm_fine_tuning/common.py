
import pandas as pd


# quiz, keyword, good_answer 의 빈칸이 채워진 DataFrame 반환
# Create Date : 2025.07.12
# Last Update Date : -

# Arguments:
# - csv_path (str) : DataFrame 이 있는 csv 파일의 path

# Returns
# - filled_dataset_df (Pandas DataFrame) : 빈칸이 채워진 DataFrame

def convert_into_filled_df(csv_path):
    dataset_df = pd.read_csv(csv_path)

    # get current columns
    data_type_list = dataset_df['data_type'].tolist()
    quiz_list = dataset_df['quiz'].tolist()
    good_answer_list = dataset_df['good_answer'].tolist()
    user_answer_list = dataset_df['user_answer'].tolist()
    explanation_list = dataset_df['explanation'].tolist()

    current_quiz = ''
    current_good_answer = ''

    # fill quiz and good answer list
    filled_quiz_list = []
    filled_good_answer_list = []
    input_data = []

    for quiz, good_answer in zip(quiz_list, good_answer_list):
        if not pd.isna(quiz) and quiz != '':
            current_quiz = quiz
        filled_quiz_list.append(current_quiz)

        if not pd.isna(good_answer) and good_answer != '':
            current_good_answer = good_answer
        filled_good_answer_list.append(current_good_answer)

        input_data.append(f'(퀴즈 문제) {current_quiz} (모범 답안) {current_good_answer}')

    # create final filled DataFrame
    filled_dataset_dict = {
        'data_type': data_type_list,
        'quiz': filled_quiz_list,
        'good_answer': filled_good_answer_list,
        'user_answer': user_answer_list,
        'explanation': explanation_list,
        'input_data': input_data
    }

    filled_dataset_df = pd.DataFrame(filled_dataset_dict)
    return filled_dataset_df
