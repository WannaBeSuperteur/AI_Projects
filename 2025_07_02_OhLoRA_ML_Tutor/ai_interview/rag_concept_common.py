
import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# AI Interview LLM 을 위한 학습된 RAG S-BERT (Sentence BERT) 모델을 이용하여 각 candidate에 대한 inference 실시 (vector 대상)
# Create Date : 2025.09.23
# Last Update Date : -

# Arguments:
# - sbert_model               (S-BERT Model)  : 학습된 Sentence BERT 모델
# - sbert_input               (str)           : S-BERT 모델에 전달할 최종 입력값
# - rag_retrieved_data_vector (Pandas Series) : DB에 저장된 데이터 (각 candidate) 에 대한 vector

# Returns:
# - similarity_score (float) : 학습된 S-BERT 모델이 계산한 similarity score

def run_inference_each_example_vector(sbert_model, sbert_input, rag_retrieved_data_vector):
    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    sbert_input_embedding = sbert_model.encode([sbert_input])
    similarity_score = compute_cosine_similarity(sbert_input_embedding[0], rag_retrieved_data_vector)

    return similarity_score


# AI Interview LLM 을 위한 학습된 RAG S-BERT 모델을 이용하여, candidate 중 가장 값이 큰 1개의 항목 반환
# Create Date : 2025.09.23
# Last Update Date : -

# Arguments:
# - sbert_model         (S-BERT Model) : 학습된 Sentence BERT 모델
# - sbert_input         (str)          : S-BERT 모델에 전달할 최종 입력값
# - candidates_csv_name (str)          : candidate 목록이 있는 csv 파일의 이름
# - verbose             (bool)         : 각 candidate item 에 대한 score 출력 여부

# Returns:
# - best_candidate (dict) : 가장 값이 큰 1개의 항목에 대한 정보 (keys: ['name', 'cos_sim'])

def pick_best_candidate(sbert_model, sbert_input, candidates_csv_name, verbose=False):
    candidate_files_path = f'{PROJECT_DIR_PATH}/ai_interview/dataset/{candidates_csv_name}'

    # read DB file
    candidates_csv = pd.read_csv(candidate_files_path, index_col=0)
    candidate_count = len(candidates_csv)
    candidates_list = []

    # compute similarity scores for each candidate
    for i in range(candidate_count):
        db_text = candidates_csv['db_data'].tolist()[i]
        db_vector = candidates_csv.iloc[i][1:]

        similarity_score = run_inference_each_example_vector(sbert_model, sbert_input, db_vector)
        candidates_list.append({'name': db_text, 'cos_sim': similarity_score})

        if verbose:
            print(f'line {i} -> DB: {db_text}, cosine similarity: {similarity_score:.4f}')

    # pick best DB item
    candidates_list.sort(key=lambda x: x['cos_sim'], reverse=True)
    return candidates_list[0]
