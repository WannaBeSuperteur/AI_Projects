import pandas as pd


if __name__ == '__main__':
    all_data = pd.read_csv('all_train_and_test_data.csv')
    all_data_train = all_data[all_data['data_type'] == 'train']
    len_train = len(all_data_train)

    # collect output part
    output_answer_types = sorted(list(set(all_data_train['output_answered'].dropna().tolist())))
    next_questions = sorted(list(set(all_data_train['output_next_question'].dropna().tolist())))

    # collect input part (question & user input)
    current_question_and_user_input = []

    for _, row in all_data_train.iterrows():
        current_question = row['current_question']
        user_input = row['user_input']

        if not pd.isna(current_question) and not pd.isna(user_input):
            current_question_and_user_input.append(f'{current_question} -> {user_input}')

    # combine input part & output part -> for S-BERT training
    output_answer_sbert_train_dataset = {'input_part': [], 'output_answer': []}
    next_questions_sbert_train_dataset = {'input_part': [], 'next_question': []}

    for input_part in current_question_and_user_input:
        for output_answer_type in output_answer_types:
            output_answer_sbert_train_dataset['input_part'].append(input_part)
            output_answer_sbert_train_dataset['output_answer'].append(output_answer_type)

        for next_question in next_questions:
            next_questions_sbert_train_dataset['input_part'].append(input_part)
            next_questions_sbert_train_dataset['next_question'].append(next_question)

    print(pd.DataFrame(output_answer_sbert_train_dataset))
    print(pd.DataFrame(next_questions_sbert_train_dataset))
