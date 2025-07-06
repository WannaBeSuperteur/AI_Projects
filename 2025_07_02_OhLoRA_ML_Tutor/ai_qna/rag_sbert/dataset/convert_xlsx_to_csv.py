import pandas as pd

if __name__ == '__main__':
    xlsx = pd.read_excel('train_test_dataset_new.xlsx')
    xlsx.rename(columns={'Unnamed: 0': 'idx0', 'Unnamed: 0.1': 'idx1'}, inplace=True)
    xlsx.drop(columns=['idx0', 'idx1'], inplace=True)
    xlsx.to_csv('train_test_dataset_new.csv', index=True)
