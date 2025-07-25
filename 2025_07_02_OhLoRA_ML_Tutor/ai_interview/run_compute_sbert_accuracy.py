
import os
import pandas as pd

NEXT_QUESTION_SBERT_RESULT_PATH = 'next_question_sbert/result'
OUTPUT_ANSWER_SBERT_RESULT_PATH = 'output_answer_sbert/result'


# test result detail 파일을 읽고 정확도 (accuracy) 및 오답 정보 반환
# Create Date : 2025.07.26
# Last Update Date : -

# Arguments:
# - file_path          (str) : test result detail 파일 경로
# - output_part_column (str) : test result detail 파일 내용 중 output part 에 해당하는 column name
#                              ('next_question' 또는 'output_answer')

# Returns:
# - accuracy          (float)     : test result detail 파일 기준 정확도
# - wrong_answer_info (list(str)) : test result detail 파일 기준 오답 정보

def read_file_and_compute_accuracy(file_path, output_part_column):
    test_result_df = pd.read_csv(file_path)
    input_part_distinct_list = list(set(test_result_df['input_part']))

    total, correct = len(input_part_distinct_list), 0
    wrong_answer_info = []

    for input_part_distinct in input_part_distinct_list:
        filtered_data = test_result_df[test_result_df['input_part'] == input_part_distinct]

        predicted_max_idx = filtered_data[f'predicted_{similarity_suffix}'].argmax()
        gt_max_idx = filtered_data[f'ground_truth_{similarity_suffix}'].argmax()
        predicted_max_value = filtered_data[output_part_column].tolist()[predicted_max_idx]
        gt_max_value = filtered_data[output_part_column].tolist()[gt_max_idx]

        if predicted_max_idx == gt_max_idx:
            correct += 1
        else:
            wrong_answer = f'input_part: {input_part_distinct} | pred: {predicted_max_value} | gt: {gt_max_value}'
            wrong_answer_info.append(wrong_answer)

    accuracy = round(correct / total, 4)
    return accuracy, wrong_answer_info


if __name__ == '__main__':
    paths = [NEXT_QUESTION_SBERT_RESULT_PATH, OUTPUT_ANSWER_SBERT_RESULT_PATH]
    similarity_score_column_suffices = ['similarity', 'score']
    data_column_names = ['next_question', 'output_answer']

    for dir_path, similarity_suffix, column_name in zip(paths, similarity_score_column_suffices, data_column_names):
        test_result_detail_files = list(filter(lambda x: x.startswith('test_result_'), os.listdir(dir_path)))
        accuracy_info_dict = {'file': test_result_detail_files, 'accuracy': [], 'wrong_answer_info': []}

        for file_name in test_result_detail_files:
            file_path = f'{dir_path}/{file_name}'
            accuracy, wrong_answer_info = read_file_and_compute_accuracy(file_path, column_name)

            accuracy_info_dict['accuracy'].append(accuracy)
            accuracy_info_dict['wrong_answer_info'].append(str(wrong_answer_info))

        accuracy_info_df = pd.DataFrame(accuracy_info_dict)
        accuracy_info_df.to_csv(f'{dir_path}/test_accuracy.csv')