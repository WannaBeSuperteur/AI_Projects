import pandas as pd

if __name__ == '__main__':
    dataset_csv = pd.read_csv('train_test_dataset_new.csv', index_col=0)

    train_dataset = dataset_csv[dataset_csv['data_type'] == 'train']
    train_dataset.drop(columns='repeat', inplace=True)

    test_dataset = dataset_csv[dataset_csv['data_type'] == 'test']
    test_dataset.drop(columns='repeat', inplace=True)

    train_dataset.to_csv('train_final.csv', index=False)
    test_dataset.to_csv('test_final.csv', index=False)
