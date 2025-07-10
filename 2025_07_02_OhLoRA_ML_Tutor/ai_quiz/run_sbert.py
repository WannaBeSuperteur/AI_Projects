try:
    from rag_sbert.inference_sbert import run_inference, run_inference_each_example
    from rag_sbert.load_sbert_model import load_trained_sbert_model
    from rag_sbert.train_sbert import train_sbert
except:
    from ai_qna.rag_sbert.inference_sbert import run_inference, run_inference_each_example
    from ai_qna.rag_sbert.load_sbert_model import load_trained_sbert_model
    from ai_qna.rag_sbert.train_sbert import train_sbert

import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# 사용자 답변 채점을 위한 학습된 S-BERT 모델 로딩
# Create Date : 2025.07.10
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델

def load_sbert_model():
    model_path = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/trained_sbert_model'
    sbert_model = load_trained_sbert_model(model_path)

    return sbert_model


if __name__ == '__main__':

    # load train & test dataset
    train_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_qna/rag_sbert/dataset/train_final.csv'
    train_dataset_df = pd.read_csv(train_dataset_csv_path)

    test_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_qna/rag_sbert/dataset/test_final.csv'
    test_dataset_df = pd.read_csv(test_dataset_csv_path)

    # load S-BERT Model
    try:
        sbert_model = load_sbert_model()
        print('S-BERT Model (for DB mechanism) - Load SUCCESSFUL! 👱‍♀️')

    except Exception as e:
        print(f'S-BERT Model (for DB mechanism) load failed : {e}')
        train_sbert(train_dataset_df)
        sbert_model = load_sbert_model()

    # run inference on test dataset
    run_inference(sbert_model, test_dataset_df)
