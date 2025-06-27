import pandas as pd

if __name__ == '__main__':
    csv = pd.read_csv('OhLoRA_fine_tuning_v3.csv')
    csv.to_excel('OhLoRA_fine_tuning_v3.xlsx')
