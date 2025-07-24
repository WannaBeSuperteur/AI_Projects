import pandas as pd


# S-BERT 학습 / 검증 / 테스트 데이터셋 추출
# Create Date : 2025.07.24
# Last Update Date : -

# Arguments:
# - data_type (str) : 학습 ('train') or 검증/테스트 ('valid/test')

# Returns:
# - dataset_df_answer_type   (Pandas DataFrame) : S-BERT 데이터셋 ("현재 질문 + 사용자 답변" & "사용자가 성공한 답변" 비교)
# - dataset_df_next_question (Pandas DataFrame) : S-BERT 데이터셋 ("현재 질문 + 사용자 답변" & "다음 질문" 비교)

def extract_sbert_dataset(data_type):
    all_data = pd.read_csv('all_train_and_test_data.csv')
    all_data_with_type = all_data[all_data['data_type'] == 'train']

    # collect output part
    output_answer_types = sorted(list(set(all_data_with_type['output_answered'].dropna().tolist())))
    next_questions = sorted(list(set(all_data_with_type['output_next_question'].dropna().tolist())))

    # collect input part (question & user input)
    current_question_and_user_input = []
    input_to_output_answer_type = {}
    input_to_next_question = {}

    for _, row in all_data_with_type.iterrows():
        current_question = row['current_question']
        user_input = row['user_input']

        if not pd.isna(current_question) and not pd.isna(user_input):
            input_sentence = f'{current_question} -> {user_input}'
            output_answer_type = row['output_answered']
            next_question = row['output_next_question']

            current_question_and_user_input.append(input_sentence)
            input_to_output_answer_type[input_sentence] = output_answer_type
            input_to_next_question[input_sentence] = next_question

    # combine input part & output part -> for S-BERT training
    output_answer_sbert_train_dataset = {'input_part': [], 'output_answer': [], 'similarity': []}
    next_questions_sbert_train_dataset = {'input_part': [], 'next_question': [], 'similarity': []}

    for input_part in current_question_and_user_input:
        for output_answer_type in output_answer_types:
            output_answer_sbert_train_dataset['input_part'].append(input_part)
            output_answer_sbert_train_dataset['output_answer'].append(output_answer_type)

            if input_to_output_answer_type[input_part] == output_answer_type:
                output_answer_sbert_train_dataset['similarity'].append(1.0)
            else:
                output_answer_sbert_train_dataset['similarity'].append(0.0)

        for next_question in next_questions:
            next_questions_sbert_train_dataset['input_part'].append(input_part)
            next_questions_sbert_train_dataset['next_question'].append(next_question)

            if input_to_next_question[input_part] == next_question:
                next_questions_sbert_train_dataset['similarity'].append(1.0)
            else:
                next_questions_sbert_train_dataset['similarity'].append(0.0)

    dataset_df_answer_type = pd.DataFrame(output_answer_sbert_train_dataset)
    dataset_df_next_question = pd.DataFrame(next_questions_sbert_train_dataset)

    return dataset_df_answer_type, dataset_df_next_question


if __name__ == '__main__':
    dataset_df_answer_type, dataset_df_next_question = extract_sbert_dataset('train')
    dataset_df_answer_type.to_csv('dataset_df_answer_type_train.csv', index=False)
    dataset_df_next_question.to_csv('dataset_df_next_question_train.csv', index=False)


