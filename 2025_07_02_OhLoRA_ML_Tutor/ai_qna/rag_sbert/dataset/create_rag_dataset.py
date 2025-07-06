
import pandas as pd


TRAIN_ITERS = 20
TEST_ITERS = 5


if __name__ == '__main__':
    train_test_dataset = pd.read_csv('train_test_dataset.csv', index_col=0)

    train_user_questions = train_test_dataset[train_test_dataset['data_type'] == 'train']['user_question']
    test_user_questions = train_test_dataset[train_test_dataset['data_type'] == 'test']['user_question']
    rag_retrieved_data = train_test_dataset[train_test_dataset['data_type'] == 'train']['rag_retrieved_data']

    n_train_original = len(train_user_questions)
    n_test_original = len(test_user_questions)

    train_user_questions_list = list(train_user_questions)
    test_user_questions_list = list(test_user_questions)
    rag_retrieved_data_list = list(rag_retrieved_data)

    new_train_test_dataset_dict = {'data_type': [], 'repeat': [], 'user_question': [], 'rag_retrieved_data': []}

    for i in range(20):
        new_train_test_dataset_dict['data_type'] += n_train_original * ['train']
        new_train_test_dataset_dict['repeat'] += n_train_original * [i]
        new_train_test_dataset_dict['user_question'] += train_user_questions_list[i:] + train_user_questions_list[:i]
        new_train_test_dataset_dict['rag_retrieved_data'] += rag_retrieved_data_list

    for i in range(5):
        new_train_test_dataset_dict['data_type'] += n_test_original * ['test']
        new_train_test_dataset_dict['repeat'] += n_train_original * [i]
        new_train_test_dataset_dict['user_question'] += test_user_questions_list[i:] + test_user_questions_list[:i]
        new_train_test_dataset_dict['rag_retrieved_data'] += rag_retrieved_data_list

    new_train_test_dataset_df = pd.DataFrame(new_train_test_dataset_dict)
    new_train_test_dataset_df.to_csv('train_test_dataset_new.csv')
