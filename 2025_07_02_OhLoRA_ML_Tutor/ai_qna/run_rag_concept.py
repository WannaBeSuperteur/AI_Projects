try:
    from rag_sbert.inference_sbert import run_inference, run_inference_each_example, run_inference_each_example_vector
    from rag_sbert.load_sbert_model import load_trained_sbert_model
    from rag_sbert.train_sbert import train_sbert
except:
    from ai_qna.rag_sbert.inference_sbert import run_inference, run_inference_each_example, run_inference_each_example_vector
    from ai_qna.rag_sbert.load_sbert_model import load_trained_sbert_model
    from ai_qna.rag_sbert.train_sbert import train_sbert

import pandas as pd

import time
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Q&A LLM 을 위한 RAG S-BERT 모델을 이용하여, DB에서 가장 적절한 1개의 항목 반환 (단, Cos-similarity >= 0.5 인 것들만)
# Create Date : 2025.07.06
# Last Update Date : -

# Arguments:
# - sbert_model  (S-BERT Model) : 학습된 Sentence BERT 모델
# - user_prompt  (str)          : Oh-LoRA 에게 전달할 사용자 프롬프트
# - db_file_name (str)          : DB 파일 (txt) 의 이름 (예: rag_data_text.txt)
# - threshold    (float)        : minimum cosine similarity threshold (default: 0.5)
# - verbose      (bool)         : 각 DB item 에 대한 score 출력 여부

# Returns:
# - best_db_item (str) : DB 파일에서 찾은 best DB item

def pick_best_db_item(sbert_model, user_prompt, db_file_name='rag_data_text.txt', threshold=0.5, verbose=False):
    db_file_path = f'{PROJECT_DIR_PATH}/ai_qna/rag_sbert/db/{db_file_name}'

    # read DB file
    db_file = open(db_file_path, 'r', encoding='UTF8')
    db_file_lines = db_file.readlines()
    db_item_list = []

    # compute similarity scores for each DB item
    for line_idx, line in enumerate(db_file_lines):
        if len(line.replace(' ', '')) < 3:
            continue

        db_text = line.split('\n')[0]
        similarity_score = run_inference_each_example(sbert_model, user_prompt, db_text)
        db_item_list.append({'db_text': db_text, 'cos_sim': similarity_score})

        if verbose:
            print(f'line {line_idx} -> DB: {db_text}, cosine similarity: {similarity_score:.4f}')

    # pick best DB item
    db_item_list.sort(key=lambda x: x['cos_sim'], reverse=True)

    if len(db_item_list) == 0:
        best_db_item = ''
    elif db_item_list[0]['cos_sim'] >= threshold:
        best_db_item = db_item_list[0]['db_text']
    else:
        best_db_item = ''

    return best_db_item


# Q&A LLM 을 위한 RAG S-BERT 모델을 이용하여, DB (csv 파일) 에서 가장 적절한 1개의 항목 반환 (단, Cos-similarity >= 0.5 인 것들만)
# Create Date : 2025.07.21
# Last Update Date : -

# Arguments:
# - sbert_model  (S-BERT Model) : 학습된 Sentence BERT 모델
# - user_prompt  (str)          : Oh-LoRA 에게 전달할 사용자 프롬프트
# - db_file_name (str)          : DB 파일 (csv) 의 이름 (예: rag_data_text.csv)
# - threshold    (float)        : minimum cosine similarity threshold (default: 0.5)
# - verbose      (bool)         : 각 DB item 에 대한 score 출력 여부

# Returns:
# - best_db_item (str) : DB 파일에서 찾은 best DB item

def pick_best_db_item_csv(sbert_model, user_prompt, db_file_name='rag_data_text.csv', threshold=0.5, verbose=False):
    db_file_path = f'{PROJECT_DIR_PATH}/ai_qna/rag_sbert/db/{db_file_name}'

    # read DB file
    db_csv = pd.read_csv(db_file_path, index_col=0)
    db_size = len(db_csv)
    db_item_list = []

    # compute similarity scores for each DB item
    for i in range(db_size):
        db_text = db_csv['db_data'].tolist()[i]
        db_vector = db_csv.iloc[i][1:]

        similarity_score = run_inference_each_example_vector(sbert_model, user_prompt, db_vector)
        db_item_list.append({'db_text': db_text, 'cos_sim': similarity_score})

        if verbose:
            print(f'line {i} -> DB: {db_text}, cosine similarity: {similarity_score:.4f}')

    # pick best DB item
    db_item_list.sort(key=lambda x: x['cos_sim'], reverse=True)

    if len(db_item_list) == 0:
        best_db_item = ''
    elif db_item_list[0]['cos_sim'] >= threshold:
        best_db_item = db_item_list[0]['db_text']
    else:
        best_db_item = ''

    return best_db_item


# Q&A LLM 을 위한 RAG 컨셉 Mechanism 학습된 모델 로딩
# Create Date : 2025.07.06
# Last Update Date : 2025.07.08
# - pretrained -> trained 로 수정

# Arguments:
# - 없음

# Returns:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델

def load_sbert_model():
    model_path = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/trained_sbert_model'
    sbert_model = load_trained_sbert_model(model_path)

    return sbert_model


if __name__ == '__main__':
    running_inference = False

    # load train & test dataset
    train_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_qna/rag_sbert/dataset/train_final.csv'
    train_dataset_df = pd.read_csv(train_dataset_csv_path)

    test_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_qna/rag_sbert/dataset/test_final.csv'
    test_dataset_df = pd.read_csv(test_dataset_csv_path)

    # load S-BERT Model
    try:
        sbert_model = load_sbert_model()
        print('S-BERT Model (for DB mechanism) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'S-BERT Model (for DB mechanism) load failed : {e}')
        train_sbert(train_dataset_df)
        sbert_model = load_sbert_model()

    # run inference on test dataset
    if running_inference:
        run_inference(sbert_model, test_dataset_df)

    # pick best DB by user prompt, from saved DB
    while True:
        test_user_prompt = input('\nInput user prompt (Ctrl+C to finish) : ')

        start_at = time.time()
        best_db_item = pick_best_db_item_csv(sbert_model,
                                             test_user_prompt,
                                             db_file_name='rag_data_text.csv',
                                             verbose=True)
        best_item_pick_time = time.time() - start_at

        print(f'\nBEST ITEM PICK TIME: {best_item_pick_time}')
        if best_db_item == '':
            print(f'NO BEST DB ITEM (cos-sim threshold : 0.6)')
        else:
            print(f'best DB item : {best_db_item}')
