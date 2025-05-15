import pandas as pd


def convert_summary(summary):
    try:
        return summary.replace('(오로라 답변 요약) ', '')
    except:
        return summary


if __name__ == '__main__':
    v2_csv = pd.read_csv('OhLoRA_fine_tuning_v2_1.csv', index_col=0)
    v2_csv['summary'] = v2_csv['summary'].map(lambda x: convert_summary(x))

    v2_csv.to_csv('OhLoRA_fine_tuning_v2_2.csv')
