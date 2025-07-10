import pandas as pd

if __name__ == '__main__':
    all_dataset_csv = pd.read_csv('all_train_and_test_data.csv')

    train_data = all_dataset_csv[all_dataset_csv['data_type'] == 'train']
    train_data.drop(columns='Unnamed: 0', inplace=True)

    valid_test_data = all_dataset_csv[all_dataset_csv['data_type'] == 'valid/test']
    valid_test_data.drop(columns='Unnamed: 0', inplace=True)

    train_data.to_csv('train_final.csv', index=False)
    valid_test_data.to_csv('valid_test_final.csv', index=False)
