
import pandas as pd


if __name__ == '__main__':
    all_data = pd.read_csv('all_train_and_test_data.csv')
    all_data_with_type = all_data[all_data['data_type'] == 'train']

    # collect output part
    output_answer_types = sorted(list(set(all_data_with_type['output_answered'].dropna().tolist())))
    next_questions = sorted(list(set(all_data_with_type['output_next_question'].dropna().tolist())))

    output_answer_types_str = '\n'.join(output_answer_types) + '\n'
    next_questions_str = '\n'.join(next_questions) + '\n'

    with open('candidates_answer_type.txt', 'w', encoding='utf-8') as f:
        f.write(output_answer_types_str)
        f.close()

    with open('candidates_next_question.txt', 'w', encoding='utf-8') as f:
        f.write(next_questions_str)
        f.close()
