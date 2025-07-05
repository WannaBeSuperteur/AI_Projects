import pandas as pd

if __name__ == '__main__':
    xlsx = pd.read_excel('train_test_dataset_new.xlsx')
    xlsx.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
    xlsx.to_csv('train_test_dataset_new.csv', index=True)
