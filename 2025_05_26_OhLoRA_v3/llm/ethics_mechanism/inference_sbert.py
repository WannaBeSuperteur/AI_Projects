import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

categories = ['사랑 고백/만남', '일반', '정치', '패드립']
n_categories = len(categories)


# Ethics Mechanism 을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "데이터셋 전체에 대한" inference 실시
# Create Date : 2025.06.02
# Last Update Date : 2025.06.02
# - Confusion Matrix, Prediction Table 등 테스트 결과 관련 부가 정보 추가

# Arguments:
# - sbert_model     (S-BERT Model)     : 학습된 Sentence BERT 모델
# - test_dataset_df (Pandas DataFrame) : 테스트 데이터셋

# Returns:
# - 반환값 없음
# - 테스트 결과 (성능지표 값) 출력 및 해당 결과를 llm/ethics_mechanism/test_result.csv 로 저장
# - 테스트 결과 (성능지표 값) 의 각 Category 별 예측값을 llm/ethics_mechanism/test_result_prediction_table.csv 로 저장
# - 테스트 결과 (성능지표 값) 의 Confusion Matrix 출력 및 해당 결과를 llm/ethics_mechanism/test_confusion_matrix.csv 로 저장

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
    test_result_dict = {'user_prompt': user_prompt_list,
                        'category': category_info_list,
                        'predicted_score': np.round(predicted_scores, 4),
                        'ground_truth_score': ground_truth_scores,
                        'absolute_error': np.round(absolute_errors, 4)}

    test_result_df = pd.DataFrame(test_result_dict)
    test_result_df.to_csv(f'{PROJECT_DIR_PATH}/llm/ethics_mechanism/test_result.csv',
                          index=False)

    # save prediction table
    prediction_table = convert_to_prediction_table(test_result_dict)
    prediction_table.to_csv(f'{PROJECT_DIR_PATH}/llm/ethics_mechanism/test_result_prediction_table.csv',
                            index=False)

    # save confusion matrix
    confusion_matrix = compute_confusion_matrix(test_result_dict)
    print(f'\nConfusion Matrix:\n{confusion_matrix}')
    confusion_matrix.to_csv(f'{PROJECT_DIR_PATH}/llm/ethics_mechanism/test_confusion_matrix.csv',
                            index=False)


# Test Result 를 각 Category 별 예측값 표 형식으로 수정
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# test_result_dict (dict) : 테스트 결과에 대한 Dictionary

# Arguments:
# prediction_table (Pandas DataFrame) : 테스트 결과에 대한 각 Category 별 예측값 표

def convert_to_prediction_table(test_result_dict):
    global categories, n_categories

    prediction_table_dict = {'user_prompt': [], 'category': []}
    for category in categories:
        prediction_table_dict[f'pred_{category}'] = []

    n_test_results = len(test_result_dict['user_prompt'])
    n_test_cases = n_test_results // n_categories

    for tc_idx in range(n_test_cases):
        prediction_table_dict['user_prompt'].append(test_result_dict['user_prompt'][tc_idx * n_categories])

        for ctg_idx, category in enumerate(categories):
            pred_score = test_result_dict['predicted_score'][tc_idx * n_categories + ctg_idx]
            gt_score = test_result_dict['ground_truth_score'][tc_idx * n_categories + ctg_idx]
            prediction_table_dict[f'pred_{category}'].append(pred_score)

            if gt_score == 1.0:
                prediction_table_dict['category'].append(category)

    prediction_table = pd.DataFrame(prediction_table_dict)
    return prediction_table


# Confusion Matrix 도출
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# test_result_dict (dict) : 테스트 결과에 대한 Dictionary

# Returns:
# confusion_matrix (Pandas DataFrame) : 테스트 결과에 대한 Confusion Matrix

def compute_confusion_matrix(test_result_dict):
    global categories, n_categories

    # confusion matrix format
    confusion_matrix_dict = {'pred | true': categories + ['total', 'recall']}
    for category in categories:
        confusion_matrix_dict[category] = [0 for _ in range(n_categories)] + [0, 0]
    confusion_matrix_dict['total'] = [0 for _ in range(n_categories)] + [0, '']
    confusion_matrix_dict['precision'] = [0 for _ in range(n_categories)] + ['', 0]

    # compute confusion matrix
    n_test_results = len(test_result_dict['user_prompt'])
    n_test_cases = n_test_results // n_categories

    for tc_idx in range(n_test_cases):
        max_pred = test_result_dict['predicted_score'][tc_idx * n_categories]
        max_pred_idx = 0

        max_gt = test_result_dict['ground_truth_score'][tc_idx * n_categories]
        max_gt_category = categories[0]

        for ctg_idx in range(n_categories):
            if test_result_dict['predicted_score'][tc_idx * n_categories + ctg_idx] > max_pred:
                max_pred = test_result_dict['predicted_score'][tc_idx * n_categories + ctg_idx]
                max_pred_idx = ctg_idx

            if test_result_dict['ground_truth_score'][tc_idx * n_categories + ctg_idx] > max_gt:
                max_gt = test_result_dict['ground_truth_score'][tc_idx * n_categories + ctg_idx]
                max_gt_category = categories[ctg_idx]

        num_after_add = int(confusion_matrix_dict[max_gt_category][max_pred_idx]) + 1
        confusion_matrix_dict[max_gt_category][max_pred_idx] = str(num_after_add)

    # finalize computing confusion matrix
    # 1. recall
    for category_idx, category in enumerate(categories):
        confusion_matrix_dict[category][n_categories] = sum(map(int, confusion_matrix_dict[category][:n_categories]))
        correct_cnt = int(confusion_matrix_dict[category][category_idx])
        total_cnt = int(confusion_matrix_dict[category][n_categories])
        confusion_matrix_dict[category][n_categories + 1] = round(correct_cnt / total_cnt, 4)

    # 2. total for precision & total test case count
    for category_idx in range(n_categories):
        for ctg in categories:
            confusion_matrix_dict['total'][category_idx] += int(confusion_matrix_dict[ctg][category_idx])
            confusion_matrix_dict['total'][n_categories] = n_test_cases

    # 3. precision
    for category_idx, category in enumerate(categories):
        correct_cnt = int(confusion_matrix_dict[category][category_idx])
        total_cnt = int(confusion_matrix_dict['total'][category_idx])
        confusion_matrix_dict['precision'][category_idx] = round(correct_cnt / total_cnt, 4)

    # 4. accuracy
    sum_correct = 0
    for category_idx, category in enumerate(categories):
        sum_correct += int(confusion_matrix_dict[category][category_idx])
    confusion_matrix_dict['precision'][n_categories + 1] = f'accuracy : {round(sum_correct / n_test_cases, 4)}'

    confusion_matrix = pd.DataFrame(confusion_matrix_dict)
    return confusion_matrix


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
