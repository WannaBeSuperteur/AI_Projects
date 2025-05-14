import pandas as pd


# 문장을 공백으로 split 했을 때의 단어 일치도에 따른 IoU Score 계산
# Create Date : 2025.05.14
# Last Update Date : -

# Arguments:
# - sentence0 (str) : 비교할 문장 0
# - sentence1 (str) : 비교할 문장 1

# Returns:
# - iou_score (float) : 두 문장을 공백으로 split 했을 때의 단어 일치도에 따른 IoU Score

def compute_sentence_iou(sentence0, sentence1):
    sentence0_split_set = set(sentence0.split(' '))
    sentence1_split_set = set(sentence1.split(' '))

    union_size = len(sentence0_split_set.union(sentence1_split_set))
    intersection_size = len(sentence0_split_set.intersection(sentence1_split_set))

    iou_score = intersection_size / union_size
    return iou_score


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
        for data1_idx in range(n):
            data0_memory_key = memory_list[data0_idx].split('[')[1].split(':')[0]
            data1_memory_key = memory_list[data1_idx].split('[')[1].split(':')[0]

            if data0_memory_key == data1_memory_key:
                similarity_score = 1.0
            else:
                similarity_score = compute_sentence_iou(data0_memory_key, data1_memory_key)

            dataset_df_dict['memory_0'].append(memory_list[data0_idx])
            dataset_df_dict['user_prompt_0'].append(user_prompt_list[data0_idx])
            dataset_df_dict['memory_1'].append(memory_list[data1_idx])
            dataset_df_dict['user_prompt_1'].append(user_prompt_list[data1_idx])
            dataset_df_dict['similarity_score'].append(round(similarity_score, 4))

    dataset_df = pd.DataFrame(dataset_df_dict)
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
