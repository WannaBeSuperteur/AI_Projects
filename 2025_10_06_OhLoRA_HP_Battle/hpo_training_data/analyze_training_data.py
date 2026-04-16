
import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
HPO_TRAINING_DATASET_PATH = f'{PROJECT_DIR_PATH}/hpo_training_data/test/'


# 학습 데이터 분석 (입력 데이터 각 column 과 출력 column 간 상관계수)
# Create Date: 2026.03.23
# Last Update Date: 2026.04.16
# - 데이터셋 파일명 교체 ('new.csv'로 끝나는 파일명으로)

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('mnist', 'fashion_mnist' or 'cifar_10')

def analyze_data(dataset_name):
    dataset_dfs = []
    suffix_list = ['']
    for suffix in suffix_list:
        dataset_csv_path = f'{HPO_TRAINING_DATASET_PATH}/{dataset_name}/hpo_model_train_dataset_df_new{suffix}.csv'
        dataset_df_part = pd.read_csv(dataset_csv_path)
        dataset_dfs.append(dataset_df_part)
    dataset_df = pd.concat(dataset_dfs, ignore_index=True)

    corr_csv_path = f'{PROJECT_DIR_PATH}/hpo_training_data/test/{dataset_name}/hpo_model_train_dataset_corrs.csv'

    # convert categorical columns to one-hot
    hp_activate_funcs = ['relu', 'leaky_relu']
    hp_optimizers = ['adam', 'adamw']
    hp_schedulers = ['exp_80', 'exp_90', 'exp_95', 'exp_98', 'cosine']

    for active_func in hp_activate_funcs:
        dataset_df[f'actfunc_{active_func}'] = dataset_df['hp_activation_func'].map(lambda x: 1 if x == active_func else 0)

    for optimizer in hp_optimizers:
        dataset_df[f'opt_{optimizer}'] = dataset_df['hp_optimizer'].map(lambda x: 1 if x == optimizer else 0)

    for scheduler in hp_schedulers:
        dataset_df[f'sch_{scheduler}'] = dataset_df['hp_scheduler'].map(lambda x: 1 if x == scheduler else 0)

    dataset_df.drop(columns=['hp_activation_func', 'hp_optimizer', 'hp_scheduler'], inplace=True)

    # analyze correlations
    correlations = dataset_df.corrwith(dataset_df['f1_score_macro'])
    correlations.to_csv(corr_csv_path)


if __name__ == '__main__':
    dataset_names = ['mnist', 'fashion_mnist', 'cifar_10']

    # run baseline CNN training & test for each dataset
    for dataset_name in dataset_names:
        analyze_data(dataset_name)
