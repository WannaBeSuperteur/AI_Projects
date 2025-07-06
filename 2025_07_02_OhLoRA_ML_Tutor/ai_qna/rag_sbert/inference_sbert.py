import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# Q&A LLM 의 RAG 컨셉을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "데이터셋 전체에 대한" inference 실시
# Create Date : 2025.07.06
# Last Update Date : -

# Arguments:
# - sbert_model     (S-BERT Model)     : 학습된 Sentence BERT 모델
# - test_dataset_df (Pandas DataFrame) : 테스트 데이터셋

# Returns:
# - 반환값 없음
# - 테스트 결과 (성능지표 값) 출력 및 해당 결과를 rag_sbert/result/test_result.csv 로 저장

def run_inference(sbert_model, test_dataset_df):
    n = len(test_dataset_df)

    memory_info_list = test_dataset_df['memory_0'].tolist()
    user_prompt_list = test_dataset_df['user_prompt_1'].tolist()

    ground_truth_scores = test_dataset_df['similarity_score'].tolist()
    predicted_scores = []

    # run prediction using trained S-BERT
    for memory_info, user_prompt in zip(memory_info_list, user_prompt_list):
        predicted_score = run_inference_each_example(sbert_model, memory_info, user_prompt)
        predicted_scores.append(predicted_score)

    # compute errors
    absolute_errors = [abs(predicted_scores[i] - ground_truth_scores[i]) for i in range(n)]

    mse_error = sum([(predicted_scores[i] - ground_truth_scores[i]) ** 2 for i in range(n)]) / n
    mae_error = sum(absolute_errors) / n
    corr_coef = np.corrcoef(predicted_scores, ground_truth_scores)[0][1]

    print(f'MSE error : {mse_error:.4f}')
    print(f'MAE error : {mae_error:.4f}')
    print(f'Corr-coef : {corr_coef:.4f}')

    # save test result
    test_result_dict = {'memory': memory_info_list, 'user_prompt': user_prompt_list,
                        'predicted_score': predicted_scores, 'ground_truth_score': ground_truth_scores,
                        'absolute_error': absolute_errors}

    test_result_df = pd.DataFrame(test_result_dict)
    result_dir = f'{PROJECT_DIR_PATH}/rag_sbert/result'
    os.makedirs(result_dir, exist_ok=True)
    test_result_df.to_csv(f'{result_dir}/test_result.csv', index=False)


# Q&A LLM 의 RAG 컨셉을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "각 example 에 대한" inference 실시
# Create Date : 2025.07.06
# Last Update Date : -

# Arguments:
# - sbert_model        (S-BERT Model) : 학습된 Sentence BERT 모델
# - user_question      (str)          : 사용자 질문 (예: "머신러닝에서 많이 쓰이는 평가지표 알려줘")
# - rag_retrieved_data (str)          : DB에 저장된 데이터 (예: "머신러닝 모델 성능 평가 지표 : Accuracy (정확도), Recall, ...")

# Returns:
# - similarity_score (float) : 학습된 S-BERT 모델이 계산한 similarity score (RAG 유사 메커니즘 용)

def run_inference_each_example(sbert_model, user_question, rag_retrieved_data):
    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    user_question_embedding = sbert_model.encode([user_question])
    rag_retrieved_data_embedding = sbert_model.encode([rag_retrieved_data])

    similarity_score = compute_cosine_similarity(user_question_embedding[0], rag_retrieved_data_embedding[0])
    return similarity_score
