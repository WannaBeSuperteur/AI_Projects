from memory_mechanism.train_sbert import train_sbert
from memory_mechanism.inference_sbert import run_inference

import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Memory Mechanism 학습된 모델 로딩
# Create Date : 2025.04.23
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델

def load_sbert_model():
    raise NotImplementedError


if __name__ == '__main__':

    # load train & test dataset
    train_dataset_csv_path = f'{PROJECT_DIR_PATH}/llm/memory_mechanism/train_dataset.csv'
    train_dataset_df = pd.read_csv(train_dataset_csv_path)

    test_dataset_csv_path = f'{PROJECT_DIR_PATH}/llm/memory_mechanism/test_dataset.csv'
    test_dataset_df = pd.read_csv(test_dataset_csv_path)

    # try load S-BERT Model -> when failed, run training and save S-BERT Model
    try:
        sbert_model = load_sbert_model()
        print('S-BERT Model (for memory mechanism) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'S-BERT Model (for memory mechanism) load failed : {e}')

        train_sbert(train_dataset_df)
        sbert_model = load_sbert_model()

    # run inference on test dataset
    run_inference(sbert_model, test_dataset_df)
