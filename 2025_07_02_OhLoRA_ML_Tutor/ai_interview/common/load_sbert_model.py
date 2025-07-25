from sentence_transformers import SentenceTransformer, models


# Pre-trained (or Fine-Tuned) S-BERT Model 로딩
# Reference : https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-라이브러리를-활용한-SBERT-학습-방법
# Create Date : 2025.07.25
# Last Update Date : -

# Arguments:
# - model_path (str) : Pre-trained (or Fine-Tuned) S-BERT Model 의 경로

# Returns:
# - trained_sbert_model (S-BERT Model) : Pre-train 된 Sentence-BERT 모델

def load_trained_sbert_model(model_path):
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
