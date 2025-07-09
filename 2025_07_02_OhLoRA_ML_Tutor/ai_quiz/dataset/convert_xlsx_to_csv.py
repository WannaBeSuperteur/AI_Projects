import pandas as pd

if __name__ == '__main__':
    xlsx = pd.read_excel('all_train_and_test_data.xlsx')
    xlsx.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
    xlsx.to_csv('all_train_and_test_data.csv', index=True)
