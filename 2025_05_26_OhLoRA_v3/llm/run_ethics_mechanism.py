
try:
    from ethics_mechanism.train_sbert import train_sbert, load_pretrained_sbert_model
    from ethics_mechanism.inference_sbert import run_inference, run_inference_each_example
except:
    from llm.ethics_mechanism.train_sbert import train_sbert, load_pretrained_sbert_model
    from llm.ethics_mechanism.inference_sbert import run_inference, run_inference_each_example

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


# Ethics Mechanism 학습된 모델 로딩
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델

def load_sbert_model():
    model_path = f'{PROJECT_DIR_PATH}/llm/models/ethics_sbert/trained_sbert_model'
    sbert_model = load_pretrained_sbert_model(model_path)

    return sbert_model


if __name__ == '__main__':

    # load train & test dataset
    train_and_test_dataset_csv_path = f'{PROJECT_DIR_PATH}/llm/ethics_mechanism/train_test_dataset.csv'
    train_and_test_dataset_df = pd.read_csv(train_and_test_dataset_csv_path)

    train_dataset_df = train_and_test_dataset_df[train_and_test_dataset_df['type'] == 'train']
    test_dataset_df = train_and_test_dataset_df[train_and_test_dataset_df['type'] == 'test']

    train_dataset_df_converted = convert_to_sbert_dataset(train_dataset_df)
    test_dataset_df_converted = convert_to_sbert_dataset(test_dataset_df)

    # try load S-BERT Model -> when failed, run training and save S-BERT Model
    try:
        sbert_model = load_sbert_model()
        print('S-BERT Model (for ethics mechanism) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'S-BERT Model (for ethics mechanism) load failed : {e}')

        train_sbert(train_dataset_df_converted)
        sbert_model = load_sbert_model()

    # run inference on test dataset
    run_inference(sbert_model, test_dataset_df_converted)
