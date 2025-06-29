
import numpy as np

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Memory Mechanism 을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "각 example 에 대한" inference 실시
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델
# - memory_info (str)          : 메모리 정보 (예: "[오늘 일정: 친구와 저녁 식사]")
# - user_prompt (str)          : Oh-LoRA LLM 에 전달되는 사용자 프롬프트

# Returns:
# - similarity_score (float) : 학습된 S-BERT 모델이 계산한 similarity score (RAG 유사 메커니즘 용)

def run_inference_each_example(sbert_model, memory_info, user_prompt):
    def compute_cosine_similarity(vector0, vector1):
        return np.dot(vector0, vector1) / (np.linalg.norm(vector0) * np.linalg.norm(vector1))

    memory_info_embedding = sbert_model.encode([memory_info])
    user_prompt_embedding = sbert_model.encode([user_prompt])

    similarity_score = compute_cosine_similarity(memory_info_embedding[0], user_prompt_embedding[0])
    return similarity_score


# Memory Mechanism 학습된 모델을 이용하여, saved memory 중 가장 적절한 1개의 메모리를 반환 (단, Cos-similarity >= 0.6 인 것들만)
# Create Date : 2025.06.29
# Last Update Date : -

# Arguments:
# - sbert_model      (S-BERT Model) : 학습된 Sentence BERT 모델
# - user_prompt      (str)          : Oh-LoRA 에게 전달할 사용자 프롬프트
# - memory_file_name (str)          : 메모리 파일 (txt) 의 이름 (예: test.txt)
# - threshold        (float)        : minimum cosine similarity threshold (default: 0.6)
# - verbose          (bool)         : 각 memory item 에 대한 score 출력 여부

# Returns:
# - best_memory (str) : 메모리 파일에서 찾은 best memory

def pick_best_memory_item(sbert_model, user_prompt, memory_file_name='test.txt', threshold=0.6, verbose=False):
    memory_file_path = f'{PROJECT_DIR_PATH}/llm/memory_mechanism/saved_memory/{memory_file_name}'

    # read memory file
    memory_file = open(memory_file_path, 'r', encoding='UTF8')
    memory_file_lines = memory_file.readlines()
    memory_item_list = []

    # compute similarity scores for each memory item
    for line_idx, line in enumerate(memory_file_lines):
        if len(line.replace(' ', '')) < 3:
            continue

        memory_text = line.split('\n')[0]
        similarity_score = run_inference_each_example(sbert_model, memory_text, user_prompt)
        memory_item_list.append({'memory_text': memory_text, 'cos_sim': similarity_score})

        if verbose:
            print(f'line {line_idx} -> memory: {memory_text}, cosine similarity: {similarity_score:.4f}')

    # pick best memory item
    memory_item_list.sort(key=lambda x: x['cos_sim'], reverse=True)

    if len(memory_item_list) == 0:
        best_memory = ''
    elif memory_item_list[0]['cos_sim'] >= threshold:
        best_memory = memory_item_list[0]['memory_text']
    else:
        best_memory = ''

    return best_memory

