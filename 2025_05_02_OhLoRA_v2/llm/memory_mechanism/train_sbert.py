from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from torch.utils.data import Dataset, DataLoader, random_split
import os

from memory_mechanism.load_sbert_model import load_pretrained_sbert_model


# remove warning "UserWarning: PyTorch is not compiled with NCCL support"
import warnings
warnings.filterwarnings('ignore')


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

SBERT_TRAIN_BATCH_SIZE = 16
SBERT_VALID_BATCH_SIZE = 4
SBERT_EPOCHS = 10

SBERT_MODEL_SAVE_PATH = f'{PROJECT_DIR_PATH}/llm/models/memory_sbert/trained_sbert_model'
SBERT_MODEL_CKPT_PATH = f'{PROJECT_DIR_PATH}/llm/models/memory_sbert/checkpoints'


class MemorySBERTDataset(Dataset):
    def __init__(self, dataset_df):
        self.memory_info = dataset_df['memory_0'].tolist()
        self.user_prompt_info = dataset_df['user_prompt_1'].tolist()
        self.similarity_score_info = dataset_df['similarity_score'].tolist()

    def __len__(self):
        return len(self.memory_info)

    def __getitem__(self, idx):
        memory_str = self.memory_info[idx]
        user_prompt = self.user_prompt_info[idx]
        similarity_score = self.similarity_score_info[idx]

        input_example = InputExample(texts=[memory_str, user_prompt],
                                     label=similarity_score)
        return input_example


# Memory Mechanism 을 위한 S-BERT (Sentence BERT) 모델 학습
# Reference : https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-라이브러리를-활용한-SBERT-학습-방법
# Create Date : 2025.05.14
# Last Update Date : -

# Arguments:
# - train_dataset_df (Pandas DataFrame) : S-BERT 학습을 위한 학습 데이터셋

# Returns:
# - 직접 반환되는 값 없음
# - 학습된 Sentence-BERT 모델을 llm/models/memory_sbert/trained_sbert_model 디렉토리에 저장

def train_sbert(train_dataset_df):
    n_train_examples = len(train_dataset_df)
    n_train_size = int(0.9 * n_train_examples)
    n_valid_size = n_train_examples - n_train_size

    # load pre-trained S-BERT model
    pretrained_sbert_model = load_pretrained_sbert_model()

    # train configurations
    train_loss = losses.CosineSimilarityLoss(model=pretrained_sbert_model)
    total_steps = SBERT_EPOCHS * (n_train_examples / SBERT_TRAIN_BATCH_SIZE)
    warmup_steps = int(0.1 * total_steps)  # first 10% of entire training process as warm-up

    # prepare train data & valid evaluator
    train_valid_dataset = MemorySBERTDataset(train_dataset_df)
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
