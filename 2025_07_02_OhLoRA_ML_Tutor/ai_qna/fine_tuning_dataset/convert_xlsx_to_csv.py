import pandas as pd

if __name__ == '__main__':
    xlsx = pd.read_excel('SFT_with_RAG_concept.xlsx')
    xlsx.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
    xlsx.to_csv('SFT_with_RAG_concept.csv', index=True)
