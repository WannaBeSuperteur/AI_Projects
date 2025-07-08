
from sentence_transformers import SentenceTransformer, models
import numpy as np
import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))

SBERT_DIMS = 768


def load_sbert_model():
    model_path = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/trained_sbert_model'

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

    trained_sbert_model = SentenceTransformer(modules=[embedding_model, pooling_model])
    return trained_sbert_model


if __name__ == '__main__':

    # load RAG data
    with open('rag_data_text.txt', 'r', encoding='UTF8') as f:
        rag_data = [line.replace('\n', '') for line in f.readlines()]
        f.close()

    rag_data_np = np.expand_dims(np.array(rag_data), axis=-1)

    # load trained S-BERT model
    sbert_model = load_sbert_model()
    print('S-BERT Model (for DB mechanism) - Load SUCCESSFUL! 👱‍♀️')

    # compute S-BERT embeddings
    rag_retrieved_data_embeddings = []
    for rag_retrieved_data in rag_data:
        rag_retrieved_data_embedding = sbert_model.encode([rag_retrieved_data])
        rag_retrieved_data_embeddings.append(rag_retrieved_data_embedding[0])
    rag_retrieved_data_embeddings_np = np.array(rag_retrieved_data_embeddings)

    final_data = np.concatenate([rag_data_np, rag_retrieved_data_embeddings_np], axis=1)
    final_data_df = pd.DataFrame(final_data)
    final_data_df.columns = ['db_data'] + list(range(SBERT_DIMS))

    print(final_data_df)
    final_data_df.to_csv('rag_data_text.csv')
