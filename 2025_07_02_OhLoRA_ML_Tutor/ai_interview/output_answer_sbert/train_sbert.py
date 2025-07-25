# cosine similarity 도출용 비교 대상: "현재 질문 + 사용자 답변" & "사용자가 성공한 답변"


from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from torch.utils.data import Dataset, DataLoader, random_split
import os

# to prevent force system off during S-BERT training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# remove warning "UserWarning: PyTorch is not compiled with NCCL support"
import warnings
warnings.filterwarnings('ignore')


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

SBERT_TRAIN_BATCH_SIZE = 16
SBERT_VALID_BATCH_SIZE = 4


class RagSBERTDataset(Dataset):
    def __init__(self, dataset_df):
        self.input_part_info = dataset_df['input_part'].tolist()
        self.output_answer_info = dataset_df['output_answer'].tolist()
        self.similarity_score_info = dataset_df['similarity'].tolist()

    def __len__(self):
        return len(self.input_part_info)

    def __getitem__(self, idx):
        input_part = self.input_part_info[idx]
        output_answer = self.output_answer_info[idx]
        similarity_score = self.similarity_score_info[idx]

        input_example = InputExample(texts=[input_part, output_answer],
                                     label=similarity_score)
        return input_example


# Pre-trained (or Fine-Tuned) S-BERT Model 로딩
# Reference : https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-라이브러리를-활용한-SBERT-학습-방법
# Create Date : 2025.07.25
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
# Create Date : 2025.07.25
# Last Update Date : -

# Arguments:
# - train_dataset_df (Pandas DataFrame) : S-BERT 학습을 위한 학습 데이터셋
# - model_path       (str)              : S-BERT 모델 경로
# - epochs           (int)              : S-BERT 학습 epoch 횟수

# Returns:
# - 직접 반환되는 값 없음
# - 학습된 Sentence-BERT 모델을 ai_qna/models/rag_sbert/trained_sbert_model 디렉토리에 저장

def train_sbert(train_dataset_df, model_path, epochs):
    sbert_model_save_path = f'{PROJECT_DIR_PATH}/ai_interview/models/next_question_sbert/trained_sbert_model_{epochs}'
    sbert_model_ckpt_path = f'{PROJECT_DIR_PATH}/ai_interview/models/next_question_sbert/checkpoints_{epochs}'

    n_train_examples = len(train_dataset_df)
    n_train_size = int(0.9 * n_train_examples)
    n_valid_size = n_train_examples - n_train_size

    print(f'train + valid size : {n_train_examples}')
    print(f'train         size : {n_train_size}')
    print(f'valid         size : {n_valid_size}')

    # save readme file
    os.makedirs(sbert_model_save_path, exist_ok=True)
    readme_path = f'{sbert_model_save_path}/readme_ML_interview_output_answer.txt'

    with open(readme_path, 'w') as f:
        f.write('Oh-LoRA AI Tutor ML interview "OUTPUT ANSWER" ' +
                '(current question + user answer -> successful user answer) model ' +
                '(defined: 20250725181901) (S-BERT, roberta-based)')
        f.close()

    # load pre-trained S-BERT model
    pretrained_sbert_model = load_pretrained_sbert_model(model_path)

    # train configurations
    train_loss = losses.CosineSimilarityLoss(model=pretrained_sbert_model)
    total_steps = epochs * (n_train_examples / SBERT_TRAIN_BATCH_SIZE)
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
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=sbert_model_save_path,
        checkpoint_path=sbert_model_ckpt_path
    )
