try:
    from sbert.inference_sbert import run_inference, run_inference_each_example
    from sbert.load_sbert_model import load_trained_sbert_model
    from sbert.train_sbert import train_sbert
except:
    from ai_quiz.sbert.inference_sbert import run_inference, run_inference_each_example
    from ai_quiz.sbert.load_sbert_model import load_trained_sbert_model
    from ai_quiz.sbert.train_sbert import train_sbert

import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# 사용자 답변 채점을 위한 학습된 S-BERT 모델 로딩
# Create Date : 2025.07.11
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델

def load_sbert_model():
    model_path = f'{PROJECT_DIR_PATH}/ai_quiz/models/sbert/trained_sbert_model'
    sbert_model = load_trained_sbert_model(model_path)

    return sbert_model


# quiz, keyword, good_answer 의 빈칸이 채워진 DataFrame 반환
# Create Date : 2025.07.11
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
    keyword_list = dataset_df['keywords'].tolist()
    good_answer_list = dataset_df['good_answer'].tolist()
    user_answer_list = dataset_df['user_answer'].tolist()
    explanation_list = dataset_df['explanation'].tolist()
    similarity_list = dataset_df['similarity_score'].tolist()

    current_quiz = ''
    current_keyword = ''
    current_good_answer = ''

    # fill quiz, keyword and good answer list
    filled_quiz_list = []
    filled_keyword_list = []
    filled_good_answer_list = []

    for quiz in quiz_list:
        if not pd.isna(quiz) and quiz != '':
            current_quiz = quiz
        filled_quiz_list.append(current_quiz)

    for keyword in keyword_list:
        if not pd.isna(keyword) and keyword != '':
            current_keyword = keyword
        filled_keyword_list.append(current_keyword)

    for good_answer in good_answer_list:
        if not pd.isna(good_answer) and good_answer != '':
            current_good_answer = good_answer
        filled_good_answer_list.append(current_good_answer)

    # create final filled DataFrame
    filled_dataset_dict = {
        'data_type': data_type_list,
        'quiz': filled_quiz_list,
        'keywords': filled_keyword_list,
        'good_answer': filled_good_answer_list,
        'user_answer': user_answer_list,
        'explanation': explanation_list,
        'similarity_score': similarity_list
    }

    filled_dataset_df = pd.DataFrame(filled_dataset_dict)
    return filled_dataset_df


if __name__ == '__main__':

    # load train & test dataset
    train_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_quiz/dataset/train_final.csv'
    train_dataset_df = convert_into_filled_df(train_dataset_csv_path)

    test_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_quiz/dataset/valid_test_final.csv'
    test_dataset_df = convert_into_filled_df(test_dataset_csv_path)

    model_path = 'sentence-transformers/all-mpnet-base-v2'

    # load S-BERT Model
    try:
        sbert_model = load_sbert_model()
        print('S-BERT Model (for DB mechanism) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'S-BERT Model (for DB mechanism) load failed : {e}')
        train_sbert(train_dataset_df, model_path)
        sbert_model = load_sbert_model()

    # run inference on test dataset
    run_inference(sbert_model, test_dataset_df, model_path)
