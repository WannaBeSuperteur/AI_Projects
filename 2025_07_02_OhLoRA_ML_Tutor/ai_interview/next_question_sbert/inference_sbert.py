# cosine similarity 도출용 비교 대상: "현재 질문 + 남은 답변 + 사용자 답변" & "다음 질문"


import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# ML Interview LLM 에 전달할 다음 질문 정보 생성을 위한 S-BERT (Sentence BERT) 모델을 이용하여 "데이터셋 전체에 대한" inference 실시
# Create Date : 2025.07.25
# Last Update Date : -

# Arguments:
# - sbert_model        (S-BERT Model)     : 학습된 Sentence BERT 모델
# - test_dataset_df    (Pandas DataFrame) : 테스트 데이터셋
# - model_path         (str)              : Pre-trained S-BERT Model 의 경로 (logging 목적으로 사용)
# - epochs             (int)              : S-BERT 학습 epoch 횟수
# - is_experiment_mode (bool)             : 실험 모드 여부

# Returns:
# - 반환값 없음
# - 테스트 결과 (성능지표 값) 출력 및 해당 결과를 rag_sbert/result/test_result.csv 로 저장

def run_inference(sbert_model, test_dataset_df, model_path, epochs, is_experiment_mode=False):
    n = len(test_dataset_df)

    input_part_list = test_dataset_df['input_part'].tolist()
    next_question_list = test_dataset_df['next_question'].tolist()

    ground_truth_similarity = test_dataset_df['similarity'].tolist()
    predicted_similarity = []

    # run prediction using trained S-BERT
    for input_part, next_question in zip(input_part_list, next_question_list):
        predicted_score = run_inference_each_example(sbert_model, input_part, next_question)
        predicted_similarity.append(predicted_score)

    # compute errors
    absolute_errors = [abs(predicted_similarity[i] - ground_truth_similarity[i]) for i in range(n)]

    mse_error = sum([(predicted_similarity[i] - ground_truth_similarity[i]) ** 2 for i in range(n)]) / n
    mae_error = sum(absolute_errors) / n
    corr_coef = np.corrcoef(predicted_similarity, ground_truth_similarity)[0][1]

    print(f'MSE error : {mse_error:.4f}')
    print(f'MAE error : {mae_error:.4f}')
    print(f'Corr-coef : {corr_coef:.4f}')

    # remove '/' from model path
    model_path_ = model_path.replace('/', '_')

    # save test result
    test_result_dict = {'input_part': input_part_list, 'next_question': next_question_list,
                        'predicted_similarity': predicted_similarity, 'ground_truth_similarity': ground_truth_similarity,
                        'absolute_error': absolute_errors}

    test_result_df = pd.DataFrame(test_result_dict)
    result_dir = f'{PROJECT_DIR_PATH}/ai_interview/next_question_sbert/result'
    os.makedirs(result_dir, exist_ok=True)

    if is_experiment_mode:
        test_result_df.to_csv(f'{result_dir}/test_result_{model_path_}_epoch_{epochs}.csv', index=False)
    else:
        test_result_df.to_csv(f'{result_dir}/test_result.csv', index=False)

    # save MSE, MAE error and Corr-coef
    metric_values_dict = {'mse': [mse_error], 'mae': [mae_error], 'corr': [corr_coef]}
    metric_values_df = pd.DataFrame(metric_values_dict)

    if is_experiment_mode:
        metric_values_df.to_csv(f'{result_dir}/test_metric_values_{model_path_}_epoch_{epochs}.csv', index=False)
    else:
        metric_values_df.to_csv(f'{result_dir}/test_metric_values.csv', index=False)


# ML Interview LLM 에 전달할 다음 질문 정보 생성을 위한 학습된 S-BERT 모델을 이용하여 "각 example 에 대한" inference 실시
# Create Date : 2025.07.25
# Last Update Date : -

# Arguments:
# - sbert_model   (S-BERT Model) : 학습된 Sentence BERT 모델
# - input_part    (str)          : 입력 파트 : "현재 질문 + 남은 답변 + 사용자 답변"
# - next_question (str)          : 출력 파트 : 다음 질문

# Returns:
# - similarity_score (float) : 학습된 S-BERT 모델이 계산한 similarity score (RAG 유사 메커니즘 용)

def run_inference_each_example(sbert_model, input_part, next_question):
    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    input_part_embedding = sbert_model.encode([input_part])
    next_question_embedding = sbert_model.encode([next_question])

    similarity_score = compute_cosine_similarity(input_part_embedding[0], next_question_embedding[0])
    return similarity_score


# ML Interview LLM 에 전달할 다음 질문 정보 생성을 위한 학습된 S-BERT 모델을 이용하여 "각 example 별" inference 실시 (vector 대상)
# Create Date : 2025.07.25
# Last Update Date : -

# Arguments:
# - sbert_model          (S-BERT Model)  : 학습된 Sentence BERT 모델
# - input_part           (str)           : 입력 파트 : "현재 질문 + 남은 답변 + 사용자 답변"
# - next_question_vector (Pandas Series) : 출력 파트 : "다음 질문" 에 대한 embedding vector

# Returns:
# - similarity_score (float) : 학습된 S-BERT 모델이 계산한 similarity score (RAG 유사 메커니즘 용)

def run_inference_each_example_vector(sbert_model, input_part, next_question_vector):
    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    input_part_embedding = sbert_model.encode([input_part])
    similarity_score = compute_cosine_similarity(input_part_embedding[0], next_question_vector)
    return similarity_score
