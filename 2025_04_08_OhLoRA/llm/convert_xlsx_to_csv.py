import pandas as pd

if __name__ == '__main__':
    xlsx = pd.read_excel('OhLoRA_fine_tuning.xlsx')
    xlsx.to_csv('OhLoRA_fine_tuning.csv')
