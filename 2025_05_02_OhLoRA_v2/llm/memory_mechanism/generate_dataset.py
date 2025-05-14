from load_sbert_model import load_pretrained_sbert_model

import pandas as pd
import numpy as np


pretrained_sbert_model = load_pretrained_sbert_model()


# S-BERT 모델을 이용하여 두 문장의 similarity score 계산
# Create Date : 2025.05.14
# Last Update Date : -

# Arguments:
# - sentence0 (str) : 비교할 문장 0
# - sentence1 (str) : 비교할 문장 1

# Returns:
# - similarity_score (float) : 두 문장의 similarity score

def compute_sentence_iou(sentence0, sentence1):
    global pretrained_sbert_model

    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    sentence0_embedding = pretrained_sbert_model.encode([sentence0])
    sentence1_embedding = pretrained_sbert_model.encode([sentence1])

    similarity_score = compute_cosine_similarity(sentence0_embedding[0], sentence1_embedding[0])
    return similarity_score


# Memory Mechanism (RAG 유사) 의 데이터셋 생성을 위한 조합 (combs) 텍스트 파일 해석 -> 최종 데이터셋으로 반환
# Create Date : 2025.05.14
# Last Update Date : -

# Arguments:
# - combs_lines (list(str)) : 조합 (combs) 텍스트 파일의 각 line

# Returns:
# - dataset_df (Pandas DataFrame) : 해당 조합 (combs) 으로 생성된 데이터셋
#                                   columns : ['memory', 'user_prompt', 'similarity_score']

def parse_combs_text(combs_lines):
    n = len(combs_lines)

    # parse Comb. lines first
    memory_list = []
    user_prompt_list = []

    for line in combs_lines:
        line_ = line.replace('[좋아하는 아이돌:', '[좋아하는 가수:')
        memory_list.append(line_.split('] ')[0] + ']')
        user_prompt_list.append(line_.split('] ')[1].split('\n')[0])

    # create dataset DataFrame
    dataset_df_dict = {'memory_0': [], 'user_prompt_0': [],
                       'memory_1': [], 'user_prompt_1': [], 'similarity_score': []}

    for data0_idx in range(n):
        print(f'generating dataset for idx {data0_idx} / {n} ...')

        for data1_idx in range(n):
            data0_memory_key = memory_list[data0_idx].split('[')[1].split(':')[0]
            data1_memory_key = memory_list[data1_idx].split('[')[1].split(':')[0]

            if data0_memory_key == '상태':
                data0_memory_key = memory_list[data0_idx].split('[')[1].split(':')[1]
            if data1_memory_key == '상태':
                data1_memory_key = memory_list[data1_idx].split('[')[1].split(':')[1]

            if data0_memory_key == data1_memory_key:
                similarity_score = 1.0
            else:
                similarity_score = compute_sentence_iou(data0_memory_key, data1_memory_key)

            dataset_df_dict['memory_0'].append(memory_list[data0_idx])
            dataset_df_dict['user_prompt_0'].append(user_prompt_list[data0_idx])
            dataset_df_dict['memory_1'].append(memory_list[data1_idx])
            dataset_df_dict['user_prompt_1'].append(user_prompt_list[data1_idx])
            dataset_df_dict['similarity_score'].append(similarity_score)

    dataset_df = pd.DataFrame(dataset_df_dict)

    # Similarity Score 추가 보정 및 소수점 4째 자리에서 반올림 처리
    dataset_df['similarity_score'] = dataset_df['similarity_score'].map(lambda x: max(2.6 * x - 1.6, 0.0))
    dataset_df['similarity_score'] = dataset_df['similarity_score'].map(lambda x: round(x, 4))

    return dataset_df


# Memory Mechanism (RAG 유사) 의 학습 및 테스트 데이터셋 생성
# Create Date : 2025.05.14
# Last Update Date : -

# Arguments:
# - dataset_type (str) : 'train' or 'test'

# Returns:
# - 최종 train, test 데이터셋을 csv 파일 ({train|test}_dataset.csv) 로 저장

def generate_dataset(dataset_type):
    combs_txt = open(f'{dataset_type}_dataset_combs.txt', 'r', encoding='UTF8')
    combs_lines = combs_txt.readlines()
    combs_txt.close()

    dataset_df = parse_combs_text(combs_lines)
    dataset_df.to_csv(f'{dataset_type}_dataset.csv', index=False)


if __name__ == '__main__':
    generate_dataset(dataset_type='train')
    generate_dataset(dataset_type='test')
