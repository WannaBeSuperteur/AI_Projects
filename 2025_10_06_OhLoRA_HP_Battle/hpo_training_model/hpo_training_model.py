
import pandas as pd
import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
HPO_TRAINING_DATA_PATH = f'{PROJECT_DIR_PATH}/hpo_training_data/test'
HPO_TRAINING_MODEL_PATH = f'{PROJECT_DIR_PATH}/hpo_training_model'

NUM_CLASSES = 10
EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA = 64
NUM_FEATURES_INPUT = 2 * EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA + NUM_CLASSES + 16
NUM_FEATURES_OUTPUT = 1

TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, TEST_BATCH_SIZE = 16, 4, 4
EARLY_STOPPING_ROUNDS = 10


class HPOTrainingModel(nn.Module):
    def __init__(self):
        super(HPOTrainingModel, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(NUM_FEATURES_INPUT, 1024),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Dropout(0.45)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(512, NUM_FEATURES_OUTPUT),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_final(x)
        return x


class HPOTrainingDataset(Dataset):
    def __init__(self, dataset_df, dataset_name, tvt_type):
        self.dataset_df = dataset_df
        self.dataset_name = dataset_name
        self.tvt_type = tvt_type          # 'train', 'valid' or 'test'

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        all_values = self.dataset_df.iloc[idx].to_dict()

        inputs = []
        labels = []
        for k, v in all_values.items():
            if k == 'f1_score_macro':
                labels.append(v)
            else:
                inputs.append(v)

        inputs_tensor = torch.tensor(inputs)
        labels_tensor = torch.tensor(labels)
        return inputs_tensor, labels_tensor


# 학습 데이터셋을 merge 하여 최종 데이터셋 생성
# Create Date : 2026.03.29
# Last Update Date : -

# Arguments:
# - dataset_name (str) : 데이터셋 이름 ('cifar_10', 'fashion_mnist' or 'mnist')

# Returns:
# - merged_dataset_df (Pandas DataFrame) : 해당 dataset name 에 대한 HPO model 의 최종 데이터셋

def merge_dataset_df(dataset_name):
    csv_path = f'{HPO_TRAINING_DATA_PATH}/{dataset_name}'
    csv_names = os.listdir(csv_path)
    dfs = []

    hp_activate_funcs = ['relu', 'leaky_relu']
    hp_optimizers = ['adam', 'adamw']
    hp_schedulers = ['exp_90', 'exp_95', 'exp_98', 'cosine']

    for csv_name in csv_names:
        if 'hpo_model_train_dataset_df' in csv_name:
            df_path = os.path.join(csv_path, csv_name)
            df = pd.read_csv(df_path)

            for active_func in hp_activate_funcs:
                df[f'actfunc_{active_func}'] = df['hp_activation_func'].map(
                    lambda x: 1 if x == active_func else 0)

            for optimizer in hp_optimizers:
                df[f'opt_{optimizer}'] = df['hp_optimizer'].map(lambda x: 1 if x == optimizer else 0)

            for scheduler in hp_schedulers:
                df[f'sch_{scheduler}'] = df['hp_scheduler'].map(lambda x: 1 if x == scheduler else 0)

            df.drop(columns=['hp_activation_func', 'hp_optimizer', 'hp_scheduler'], inplace=True)
            dfs.append(df)

    merged_dataset_df = pd.concat(dfs, ignore_index=True)
    return merged_dataset_df


# HPO 모델 로딩 (학습할 모델)
# Create Date : 2026.03.29
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - hpo_model (torch.nn.modules) : 학습할 HPO 모델

def load_hpo_model():
    hpo_model = HPOTrainingModel()
    hpo_model.optimizer = torch.optim.AdamW(hpo_model.parameters(), lr=0.001)
    hpo_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=hpo_model.optimizer, gamma=0.95)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hpo_model.to(device)
    hpo_model.device = device

    return hpo_model


# HPO 모델 학습 및 저장
# Create Date : 2026.04.04
# Last Update Date : -

# Arguments:
# - train_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - hpo_model     (torch.nn.modules)         : HPO 모델

def train_hpo_model(train_dataset, hpo_model):
    hpo_model.train()

    n_train_size = int(0.9 * len(train_dataset))
    n_valid_size = len(train_dataset) - n_train_size
    train_dataset_train, train_dataset_valid = random_split(train_dataset, [n_train_size, n_valid_size])

    train_loader = DataLoader(train_dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(train_dataset_valid, batch_size=VALID_BATCH_SIZE, shuffle=True)
    loss_func = nn.MSELoss(reduction='sum')

    current_epoch = 0
    min_valid_loss_epoch = -1  # Loss-based Early Stopping (Loss = Error = MSE for HPO training model)
    min_valid_loss = None
    best_epoch_model = None
    val_loss_list = []

    while True:

        # training
        hpo_model.train()
        train_total_rows = 0
        train_loss_sum = 0.0

        for idx, (inputs, labels) in enumerate(train_loader):
            inputs_ = inputs.cuda()
            labels_ = labels.cuda()

            hpo_model.optimizer.zero_grad()
            outputs = hpo_model(inputs_).to(torch.float32)

            loss = loss_func(outputs, labels_)
            loss.backward()
            hpo_model.optimizer.step()

            train_loss_sum += loss.item()
            train_total_rows += labels.size(0)

        train_loss = train_loss_sum / train_total_rows

        # validation
        hpo_model.eval()
        valid_total_rows = 0
        valid_loss_sum = 0.0

        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs_ = inputs.cuda()
            labels_ = labels.cuda()
            with torch.no_grad():
                outputs = hpo_model(inputs_).to(torch.float32)

            loss = loss_func(outputs, labels_)
            valid_loss_sum += loss.item()
            valid_total_rows += labels.size(0)

        valid_loss = valid_loss_sum / valid_total_rows
        print(f'epoch: {current_epoch}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}')
        val_loss_list.append(valid_loss)

        # handle early stopping
        current_epoch += 1
        if min_valid_loss is None or valid_loss < min_valid_loss:
            min_valid_loss_epoch = current_epoch
            best_epoch_model = load_hpo_model().to(hpo_model.device)
            best_epoch_model.device = hpo_model.device
            best_epoch_model.load_state_dict(hpo_model.state_dict())

        if current_epoch - min_valid_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

    # save best epoch model
    torch.save(best_epoch_model.state_dict(), f'{HPO_TRAINING_MODEL_PATH}/hpo_model_{dataset_name}.pt')


# HPO 모델 테스트
# Create Date : 2026.03.29
# Last Update Date : -

# Arguments:
# - test_dataset (torch.utils.data.Dataset) : 학습 (train) 데이터셋
# - hpo_model    (torch.nn.modules)         : HPO 모델

def test_hpo_model(test_dataset, hpo_model):
    raise NotImplementedError


# HPO 모델 tabular 데이터 전처리를 위한 (학습 데이터 기준) 각 column의 평균, 표준편차 계산 + 파일로 저장 (향후 inference 시 처리용)
# Create Date : 2026.03.30
# Last Update Date : -

# Arguments:
# - train_df (Pandas DataFrame) : 학습 데이터셋의 DataFrame

# Returns:
# - train_means (dict(float)) : 학습 데이터셋의 각 input column 별 평균값 목록
# - train_stds  (dict(float)) : 학습 데이터셋의 각 input column 별 표준편차 목록

def get_means_and_stds(train_df):
    train_means = train_df.mean().to_dict()
    train_stds = train_df.std().to_dict()

    return train_means, train_stds


# HPO 모델 tabular 데이터 전처리 함수
# Create Date : 2026.03.31
# Last Update Date : -

# Arguments:
# - dataset_df (Pandas DataFrame) : 학습 또는 테스트 데이터셋의 DataFrame
# - means      (dict(float))      : 각 input column 별 평균값 목록
# - stds       (dict(float))      : 각 input column 별 표준편차 목록

# Returns:
# - preprocessed_dataset_df (Pandas DataFrame) : 학습 또는 테스트 데이터셋의 전처리된 DataFrame

def preprocess_data(dataset_df, means, stds):
    target_column = 'f1_score_macro'
    column_names = list(dataset_df.columns)
    preprocessed_dataset_df = pd.DataFrame(dataset_df)

    for column in column_names:
        if column != target_column:
            preprocessed_dataset_df[column] = dataset_df[column].apply(lambda x: (x - means[column]) / stds[column])

    return preprocessed_dataset_df


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    for dataset_name in dataset_names:
        print(f'\ndataset name : {dataset_name}')

        merged_dataset_df = merge_dataset_df(dataset_name)
        merged_dataset_size = len(merged_dataset_df)
        merged_dataset_train_size = int(0.9 * merged_dataset_size)

        train_df_raw = merged_dataset_df.iloc[:merged_dataset_train_size, :]
        test_df_raw = merged_dataset_df.iloc[merged_dataset_train_size:, :]

        # pre-process train & test data
        train_means, train_stds = get_means_and_stds(train_df_raw)
        train_df = preprocess_data(train_df_raw, train_means, train_stds)
        test_df = preprocess_data(test_df_raw, train_means, train_stds)

        train_dataset = HPOTrainingDataset(dataset_df=train_df, dataset_name=dataset_name, tvt_type='train')
        test_dataset = HPOTrainingDataset(dataset_df=train_df, dataset_name=dataset_name, tvt_type='test')

        # train and test HPO model
        hpo_model = load_hpo_model()
        train_hpo_model(train_dataset, hpo_model)
        test_hpo_model(test_dataset, hpo_model)
