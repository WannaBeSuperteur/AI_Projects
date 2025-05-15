import pandas as pd


def convert_input_data(input_data):
    if input_data[0] == '<':
        input_data = input_data.replace('<', '(오로라 답변 요약) ')
        input_data = input_data.replace('>', ' (사용자 질문)')
    return input_data


def convert_summary(summary):
    try:
        if summary[0] == '<':
            return summary.replace('<', '(오로라 답변 요약) ').replace('>', '')
    except:
        return summary


if __name__ == '__main__':
    v2_csv = pd.read_csv('OhLoRA_fine_tuning_v2.csv', index_col=0)
    v2_csv['input_data'] = v2_csv['input_data'].map(lambda x: convert_input_data(x))
    v2_csv['summary'] = v2_csv['summary'].map(lambda x: convert_summary(x))

    v2_csv.to_csv('OhLoRA_fine_tuning_v2_1.csv')
