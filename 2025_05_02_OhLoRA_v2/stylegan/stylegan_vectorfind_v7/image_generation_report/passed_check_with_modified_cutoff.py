import pandas as pd


def check_passed(df_row, cutoff):
    eyes_passed = abs(df_row['eyes_corr']) >= cutoff['eyes']
    mouth_passed = abs(df_row['mouth_corr']) >= cutoff['mouth']
    pose_passed = abs(df_row['pose_corr']) >= cutoff['pose']
    passed = eyes_passed and mouth_passed and pose_passed

    return 'O' if passed else 'X'


if __name__ == '__main__':
    cutoff = {'eyes': 0.9, 'mouth': 0.9, 'pose': 0.88}

    test_result_csv = pd.read_csv('test_result.csv', index_col=0)
    test_result_csv['passed_rechecked'] = test_result_csv.apply(lambda x: check_passed(x, cutoff), axis=1)
    test_result_csv.to_csv('test_result.csv')

    # print passed idx & passed count (with modified cutoff)
    passed_count = 0

    for i in range(len(test_result_csv)):
        if test_result_csv['passed_rechecked'][i] == 'O':
            print(f'passed idx : {i}')
            passed_count += 1

    print(f'rechecked passed count : {passed_count}')
