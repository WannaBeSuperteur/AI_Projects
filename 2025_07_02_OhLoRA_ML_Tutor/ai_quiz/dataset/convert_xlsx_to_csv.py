import pandas as pd

if __name__ == '__main__':
    xlsx = pd.read_excel('question_list.xlsx')
    xlsx.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
    xlsx.to_csv('question_list.csv', index=True)
