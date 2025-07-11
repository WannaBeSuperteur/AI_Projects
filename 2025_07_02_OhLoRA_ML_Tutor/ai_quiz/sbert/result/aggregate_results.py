
import os
import pandas as pd


if __name__ == '__main__':
    test_result_csvs = list(filter(lambda x: x.startswith('test_metric_values'), os.listdir()))
    test_result_aggregate_dict = {'model': [], 'epochs': [], 'mse': [], 'mae': [], 'corr': []}

    for csv in test_result_csvs:
        model_name = csv.split('_values_')[1].split('_epoch_')[0]
        epochs = int(csv.split('_epoch_')[1].split('.')[0])

        df = pd.read_csv(csv)
        mse = round(df['mse'][0], 4)
        mae = round(df['mae'][0], 4)
        corr = round(df['corr'][0], 4)

        test_result_aggregate_dict['model'].append(model_name)
        test_result_aggregate_dict['epochs'].append(epochs)
        test_result_aggregate_dict['mse'].append(mse)
        test_result_aggregate_dict['mae'].append(mae)
        test_result_aggregate_dict['corr'].append(corr)

    test_result_aggregate_df = pd.DataFrame(test_result_aggregate_dict)
    test_result_aggregate_df.sort_values(by=['model', 'epochs'], inplace=True)

    test_result_aggregate_df.to_csv('aggregrated.csv', index=False)
