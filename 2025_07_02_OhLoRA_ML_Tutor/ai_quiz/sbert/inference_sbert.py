import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# 사용자 답변 채점을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "데이터셋 전체에 대한" inference 실시
# Create Date : 2025.07.11
# Last Update Date : -

# Arguments:
# - sbert_model     (S-BERT Model)     : 학습된 Sentence BERT 모델
# - test_dataset_df (Pandas DataFrame) : 테스트 데이터셋
# - model_path      (str)              : Pre-trained S-BERT Model 의 경로 (logging 목적으로 사용)

# Returns:
# - 반환값 없음
# - 테스트 결과 (성능지표 값) 출력 및 해당 결과를 sbert/result/test_result.csv 로 저장

def run_inference(sbert_model, test_dataset_df, model_path):
    n = len(test_dataset_df)

    user_answer_list = test_dataset_df['user_answer'].tolist()
    good_answer_list = test_dataset_df['good_answer'].tolist()
    ground_truth_scores = test_dataset_df['similarity_score'].tolist()
    predicted_scores = []

    # run prediction using trained S-BERT
    for user_answer, good_answer in zip(user_answer_list, good_answer_list):
        predicted_score = run_inference_each_example(sbert_model, user_answer, good_answer)
        predicted_scores.append(predicted_score)

    # compute errors
    absolute_errors = [abs(predicted_scores[i] - ground_truth_scores[i]) for i in range(n)]

    mse_error = sum([(predicted_scores[i] - ground_truth_scores[i]) ** 2 for i in range(n)]) / n
    mae_error = sum(absolute_errors) / n
    corr_coef = np.corrcoef(predicted_scores, ground_truth_scores)[0][1]

    print(f'MSE error : {mse_error:.4f}')
    print(f'MAE error : {mae_error:.4f}')
    print(f'Corr-coef : {corr_coef:.4f}')

    # remove '/' from model path
    model_path_ = model_path.replace('/', '_')

    # save test result
    test_result_dict = {'user_answer': user_answer_list, 'good_answer': good_answer_list,
                        'predicted_score': predicted_scores, 'ground_truth_score': ground_truth_scores,
                        'absolute_error': absolute_errors}

    test_result_df = pd.DataFrame(test_result_dict)
    result_dir = f'{PROJECT_DIR_PATH}/ai_quiz/sbert/result'
    os.makedirs(result_dir, exist_ok=True)
    test_result_df.to_csv(f'{result_dir}/test_result_{model_path_}.csv', index=False)

    # save MSE, MAE error and Corr-coef
    metric_values_dict = {'mse': [mse_error], 'mae': [mae_error], 'corr': [corr_coef]}
    metric_values_df = pd.DataFrame(metric_values_dict)
    metric_values_df.to_csv(f'{result_dir}/test_metric_values_{model_path_}.csv', index=False)


# 사용자 답변 채점을 위한 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "각 example 에 대한" inference 실시
# Create Date : 2025.07.11
# Last Update Date : -

# Arguments:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델
# - user_answer (str)          : 사용자 답변 (예: "코사인 유사도는 벡터의 방향을 ...")
# - good_answer (str)          : 모범 답안 (예: "Cosine Similarity (코사인 유사도) 는 벡터의 크기가 아닌 방향을 ...")

# Returns:
# - similarity_score (float) : 학습된 S-BERT 모델이 계산한 similarity score

def run_inference_each_example(sbert_model, user_answer, good_answer):
    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    user_answer_embedding = sbert_model.encode([user_answer])
    good_answer_embedding = sbert_model.encode([good_answer])

    similarity_score = compute_cosine_similarity(user_answer_embedding[0], good_answer_embedding[0])
    return similarity_score
