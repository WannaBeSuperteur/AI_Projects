
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

categories = ['사랑 고백/만남', '일반', '정치', '패드립']


# 원본 데이터셋을 S-BERT 모델로 직접 학습 가능한 데이터셋으로 변환
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# dataset_df (Pandas DataFrame) : 데이터셋의 Pandas DataFrame
#                                 columns = ['type', 'user_prompt', 'category']

# Returns:
# converted_df (Pandas DataFrame) : S-BERT 가 직접 학습 가능하도록 변환된 DataFrame
#                                   columns = ['type', 'user_prompt', 'category', 'ground_truth_score']

def convert_to_sbert_dataset(dataset_df):
    converted_df_dict = {'type': [], 'user_prompt': [], 'category': [], 'ground_truth_score': []}

    for idx, row in dataset_df.iterrows():
        for category in categories:
            converted_df_dict['type'].append(row['type'])
            converted_df_dict['user_prompt'].append(row['user_prompt'])
            converted_df_dict['category'].append(category)

            if category == row['category']:
                converted_df_dict['ground_truth_score'].append(1.0)
            else:
                converted_df_dict['ground_truth_score'].append(0.0)

    converted_df = pd.DataFrame(converted_df_dict)
    return converted_df


if __name__ == '__main__':

    # load train & test dataset
    train_and_test_dataset_csv_path = f'{PROJECT_DIR_PATH}/llm/ethics_mechanism/train_test_dataset.csv'
    train_and_test_dataset_df = pd.read_csv(train_and_test_dataset_csv_path)

    train_dataset_df = train_and_test_dataset_df[train_and_test_dataset_df['type'] == 'train']
    test_dataset_df = train_and_test_dataset_df[train_and_test_dataset_df['type'] == 'test']

    train_dataset_df_converted = convert_to_sbert_dataset(train_dataset_df)
    test_dataset_df_converted = convert_to_sbert_dataset(test_dataset_df)
