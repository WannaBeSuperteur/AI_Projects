from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from torch.utils.data import Dataset, DataLoader, random_split
import os

# remove warning "UserWarning: PyTorch is not compiled with NCCL support"
import warnings
warnings.filterwarnings('ignore')


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

SBERT_TRAIN_BATCH_SIZE = 16
SBERT_VALID_BATCH_SIZE = 4
SBERT_EPOCHS = 10

SBERT_MODEL_SAVE_PATH = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/trained_sbert_model'
SBERT_MODEL_CKPT_PATH = f'{PROJECT_DIR_PATH}/ai_qna/models/rag_sbert/checkpoints'


class RagSBERTDataset(Dataset):
    def __init__(self, dataset_df):
        self.user_question_info = dataset_df['user_question'].tolist()
        self.rag_retrieved_data_info = dataset_df['rag_retrieved_data'].tolist()
        self.similarity_score_info = dataset_df['similarity_score'].tolist()

    def __len__(self):
        return len(self.user_question_info)

    def __getitem__(self, idx):
        user_question = self.user_question_info[idx]
        rag_retrieved_data = self.rag_retrieved_data_info[idx]
        similarity_score = self.similarity_score_info[idx]

        input_example = InputExample(texts=[user_question, rag_retrieved_data],
                                     label=similarity_score)
        return input_example


# Pre-trained (or Fine-Tuned) S-BERT Model 로딩
# Reference : https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-라이브러리를-활용한-SBERT-학습-방법
# Create Date : 2025.07.06
# Last Update Date : -

# Arguments:
# - model_path (str) : Pre-trained (or Fine-Tuned) S-BERT Model 의 경로

# Returns:
# - pretrained_sbert_model (S-BERT Model) : Pre-train 된 Sentence-BERT 모델

def load_pretrained_sbert_model(model_path="klue/roberta-base"):
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


# Q&A LLM RAG 컨셉 Mechanism 을 위한 S-BERT (Sentence BERT) 모델 학습
# Reference : https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-라이브러리를-활용한-SBERT-학습-방법
# Create Date : 2025.07.06
# Last Update Date : -

# Arguments:
# - train_dataset_df (Pandas DataFrame) : S-BERT 학습을 위한 학습 데이터셋

# Returns:
# - 직접 반환되는 값 없음
# - 학습된 Sentence-BERT 모델을 ai_qna/models/rag_sbert/trained_sbert_model 디렉토리에 저장

def train_sbert(train_dataset_df):
    n_train_examples = len(train_dataset_df)
    n_train_size = int(0.9 * n_train_examples)
    n_valid_size = n_train_examples - n_train_size

    print(f'train + valid size : {n_train_examples}')
    print(f'train         size : {n_train_size}')
    print(f'valid         size : {n_valid_size}')

    # save readme file
    os.makedirs(SBERT_MODEL_SAVE_PATH, exist_ok=True)
    readme_path = f'{SBERT_MODEL_SAVE_PATH}/readme_QNA.txt'

    with open(readme_path, 'w') as f:
        f.write('Oh-LoRA AI Tutor Q&A model (defined: 20250706132519) (S-BERT, roberta-based)')
        f.close()

    # load pre-trained S-BERT model
    pretrained_sbert_model = load_pretrained_sbert_model()

    # train configurations
    train_loss = losses.CosineSimilarityLoss(model=pretrained_sbert_model)
    total_steps = SBERT_EPOCHS * (n_train_examples / SBERT_TRAIN_BATCH_SIZE)
    warmup_steps = int(0.1 * total_steps)  # first 10% of entire training process as warm-up

    # prepare train data & valid evaluator
    train_valid_dataset = RagSBERTDataset(train_dataset_df)
    train_dataset, valid_dataset = random_split(train_valid_dataset, [n_train_size, n_valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=SBERT_TRAIN_BATCH_SIZE, shuffle=True)
    valid_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        valid_dataset,
        name="valid_evaluator"
    )

    # run training
    evaluation_steps = len(train_dataloader) - int(len(train_dataloader) * 0.9)

    pretrained_sbert_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=valid_evaluator,
        epochs=SBERT_EPOCHS,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=SBERT_MODEL_SAVE_PATH,
        checkpoint_path=SBERT_MODEL_CKPT_PATH
    )
