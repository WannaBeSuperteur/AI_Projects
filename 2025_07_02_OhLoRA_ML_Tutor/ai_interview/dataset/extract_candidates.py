
from sentence_transformers import SentenceTransformer, models

import numpy as np
import pandas as pd
import os

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
SBERT_DIMS = 768


def load_pretrained_sbert_model(model_path):
    embedding_model = models.Transformer(
        model_name_or_path=model_path,
        max_seq_length=64,
        do_lower_case=True
    )

    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    pretrained_sbert_model = SentenceTransformer(modules=[embedding_model, pooling_model])
    return pretrained_sbert_model


def extract_sbert_embeddings(output_part_name, output_part_list, sbert_model_path):
    output_part_list_np = np.expand_dims(np.array(output_part_list), axis=-1)

    # load trained S-BERT model
    sbert_model = load_pretrained_sbert_model(f'{PROJECT_DIR_PATH}/ai_interview/models/{sbert_model_path}')
    print(f'S-BERT Model ({output_part_name}) - Load SUCCESSFUL! 👱‍♀️')

    # compute S-BERT embeddings
    output_part_embeddings = []
    for output_part in output_part_list:
        output_part_embedding = sbert_model.encode([output_part])
        output_part_embeddings.append(output_part_embedding[0])
    output_part_embeddings_np = np.array(output_part_embeddings)

    final_data = np.concatenate([output_part_list_np, output_part_embeddings_np], axis=1)
    final_data_df = pd.DataFrame(final_data)
    final_data_df.columns = ['db_data'] + list(range(SBERT_DIMS))

    final_data_df.to_csv(f'embeddings_{output_part_name}.csv')


if __name__ == '__main__':
    all_data = pd.read_csv('all_train_and_test_data.csv')
    all_data_with_type = all_data[all_data['data_type'] == 'train']

    # collect output part
    output_answer_types = sorted(list(set(all_data_with_type['output_answered'].dropna().tolist())))
    next_questions = sorted(list(set(all_data_with_type['output_next_question'].dropna().tolist())))

    output_answer_types_str = '\n'.join(output_answer_types) + '\n'
    next_questions_str = '\n'.join(next_questions) + '\n'

    with open('candidates_answer_type.txt', 'w', encoding='utf-8') as f:
        f.write(output_answer_types_str)
        f.close()

    with open('candidates_next_question.txt', 'w', encoding='utf-8') as f:
        f.write(next_questions_str)
        f.close()

    extract_sbert_embeddings(output_part_name='answer_type',
                             output_part_list=output_answer_types,
                             sbert_model_path='output_answer_sbert/trained_sbert_model_40')

    extract_sbert_embeddings(output_part_name='next_question',
                             output_part_list=next_questions,
                             sbert_model_path='next_question_sbert/trained_sbert_model_40')
