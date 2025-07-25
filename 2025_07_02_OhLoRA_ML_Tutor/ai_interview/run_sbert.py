
try:
    from common.load_sbert_model import load_trained_sbert_model
    from next_question_sbert.inference_sbert import run_inference as run_inference_next_question
    from next_question_sbert.train_sbert import train_sbert as train_sbert_next_question
    from output_answer_sbert.inference_sbert import run_inference as run_inference_output_answer
    from output_answer_sbert.train_sbert import train_sbert  as train_sbert_output_answer

except:
    from ai_interview.common.load_sbert_model import load_trained_sbert_model
    from ai_interview.next_question_sbert.inference_sbert import run_inference_next_question
    from ai_interview.next_question_sbert.train_sbert import train_sbert_next_question
    from ai_interview.output_answer_sbert.inference_sbert import run_inference_output_answer
    from ai_interview.output_answer_sbert.train_sbert import train_sbert_output_answer

import pandas as pd

import os
import shutil
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# 다음 질문 또는 사용자가 성공한 답변 예측용 S-BERT 모델 로딩
# Create Date : 2025.07.25
# Last Update Date : -

# Arguments:
# - model_type (str) : 다음 질문 ('next_question') or 사용자가 성공한 답변 ('output_answer') 예측용 모델
# - epochs     (int) : S-BERT 학습 epoch 횟수

# Returns:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델

def load_sbert_model(model_type, epochs):
    assert model_type in ['next_question', 'output_answer'], "model_type must be 'next_question' or 'output_answer'."

    model_path = f'{PROJECT_DIR_PATH}/ai_interview/models/{model_type}_sbert/trained_sbert_model_{epochs}'
    sbert_model = load_trained_sbert_model(model_path)

    return sbert_model


# 다음 질문 또는 사용자가 성공한 답변 예측용 S-BERT 모델 학습
# Create Date : 2025.07.25
# Last Update Date : -

# Arguments:
# - model_type      (str)      : 다음 질문 ('next_question') or 사용자가 성공한 답변 ('output_answer') 예측용 모델
# - experiment_mode (boolean)  : 실험 모드 여부
# - train_sbert     (function) : S-BERT 모델 학습 함수
# - run_inference   (function) : S-BERT 모델 inference 실행 함수

# Returns:
# - 해당 model_type 의 S-BERT 모델 학습
# - 해당 S-BERT 모델에 대한 inference test 실시 및 그 결과 저장

def run_sbert_each_model(model_type, experiment_mode, train_sbert, run_inference):
    assert model_type in ['next_question', 'output_answer'], "model_type must be 'next_question' or 'output_answer'."

    # load train & test dataset
    if model_type == 'next_question':
        dataset_symbol = 'next_question'
    else:  # output_answer
        dataset_symbol = 'answer'

    train_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_interview/dataset/dataset_df_{dataset_symbol}_train.csv'
    test_dataset_csv_path = f'{PROJECT_DIR_PATH}/ai_interview/dataset/dataset_df_{dataset_symbol}_valid_test.csv'
    train_dataset_df = pd.read_csv(train_dataset_csv_path)
    test_dataset_df = pd.read_csv(test_dataset_csv_path)

    # experiment mode
    if experiment_mode:
        model_path_list = ['klue/roberta-base']
        epochs_list = [5, 10, 20, 40]

        for model_path in model_path_list:
            for epochs in epochs_list:
                train_sbert(train_dataset_df, model_path, epochs)
                sbert_model = load_sbert_model(model_type, epochs)
                run_inference(sbert_model, test_dataset_df, model_path, epochs, is_experiment_mode=True)

                models_dir = f'{PROJECT_DIR_PATH}/ai_interview/models'
                shutil.rmtree(models_dir)

    # NOT experiment mode
    else:

        # final decision
        model_path = 'klue/roberta-base'
        epochs = 100

        # load S-BERT Model
        try:
            sbert_model = load_sbert_model(model_type, epochs)
            print('S-BERT Model (for DB mechanism) - Load SUCCESSFUL! 👱‍♀️')

        except Exception as e:
            print(f'S-BERT Model (for DB mechanism) load failed : {e}')
            train_sbert(train_dataset_df, model_path, epochs)
            sbert_model = load_sbert_model(model_type, epochs)

        # run inference on test dataset
        run_inference(sbert_model, test_dataset_df, model_path, epochs)


if __name__ == '__main__':
    run_sbert_each_model(model_type='next_question',
                         experiment_mode=True,
                         train_sbert=train_sbert_next_question,
                         run_inference=train_sbert_next_question)

    run_sbert_each_model(model_type='output_answer',
                         experiment_mode=True,
                         train_sbert=train_sbert_output_answer,
                         run_inference=train_sbert_output_answer)
