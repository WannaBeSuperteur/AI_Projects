import pandas as pd

if __name__ == '__main__':
    xlsx = pd.read_excel('OhLoRA_fine_tuning_v3.xlsx')
    xlsx.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
    xlsx.to_csv('OhLoRA_fine_tuning_v3.csv', index=False)
