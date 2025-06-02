import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# Ethics Mechanism 을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "데이터셋 전체에 대한" inference 실시
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# - sbert_model     (S-BERT Model)     : 학습된 Sentence BERT 모델
# - test_dataset_df (Pandas DataFrame) : 테스트 데이터셋

# Returns:
# - 반환값 없음
# - 테스트 결과 (성능지표 값) 출력 및 해당 결과를 llm/ethics_mechanism/test_result.csv 로 저장

def run_inference(sbert_model, test_dataset_df):
    n = len(test_dataset_df)

    user_prompt_list = test_dataset_df['user_prompt'].tolist()
    category_info_list = test_dataset_df['category'].tolist()
    ground_truth_scores = test_dataset_df['ground_truth_score'].tolist()
    predicted_scores = []

    # run prediction using trained S-BERT
    for user_prompt, category_info in zip(user_prompt_list, category_info_list):
        predicted_score = run_inference_each_example(sbert_model, user_prompt, category_info)
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
    test_result_dict = {'user_prompt': user_prompt_list, 'category': category_info_list,
                        'predicted_score': predicted_scores, 'ground_truth_score': ground_truth_scores,
                        'absolute_error': absolute_errors}

    test_result_df = pd.DataFrame(test_result_dict)
    test_result_df.to_csv(f'{PROJECT_DIR_PATH}/llm/ethics_mechanism/test_result.csv', index=False)


# Ethics Mechanism 을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "각 example 에 대한" inference 실시
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# - sbert_model   (S-BERT Model) : 학습된 Sentence BERT 모델
# - user_prompt   (str)          : Oh-LoRA LLM 에 전달되는 사용자 프롬프트
# - category_info (str)          : 해당 사용자 프롬프트가 속한 카테고리 ('일반', '사랑 고백/만남', '정치', '패드립' 중 하나)

# Returns:
# - predicted_score (float) : 학습된 S-BERT 모델이 계산한, 해당 카테고리에 대한 predicted score

def run_inference_each_example(sbert_model, user_prompt, category_info):
    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    user_prompt_embedding = sbert_model.encode([user_prompt])
    category_info_embedding = sbert_model.encode([category_info])

    predicted_score = compute_cosine_similarity(user_prompt_embedding[0], category_info_embedding[0])
    return predicted_score
