import pandas as pd

if __name__ == '__main__':
    csv = pd.read_csv('train_test_dataset_new.csv')
    csv.to_excel('train_test_dataset_new.xlsx')
