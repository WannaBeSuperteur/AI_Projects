import pandas as pd


def change_time_marks(input_data):
    input_data = input_data.replace('아침) ', '오전) ')

    dows = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    time_of_days = ['오전', '오후', '저녁']

    for dow in dows:
        for time_of_day in time_of_days:
            before = f'({dow} {time_of_day})'
            after = f'(지금은 {dow} {time_of_day})'
            input_data = input_data.replace(before, after)

    return input_data


if __name__ == '__main__':
    v3_dataset = pd.read_csv('OhLoRA_fine_tuning_v3.csv')
    v3_dataset['input_data'] = v3_dataset['input_data'].apply(lambda x: change_time_marks(x))
    v3_dataset.to_csv('OhLoRA_fine_tuning_v3.csv', index=False)
