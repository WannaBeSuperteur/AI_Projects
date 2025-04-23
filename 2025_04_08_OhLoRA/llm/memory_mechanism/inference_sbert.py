

# Memory Mechanism 을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "데이터셋 전체에 대한" inference 실시
# Create Date : 2025.04.23
# Last Update Date : -

# Arguments:
# - sbert_model     (S-BERT Model)     : 학습된 Sentence BERT 모델
# - test_dataset_df (Pandas DataFrame) : 테스트 데이터셋

# Returns:
# - 반환값 없음
# - 테스트 결과 (성능지표 값) 출력됨

def run_inference(sbert_model, test_dataset_df):
    raise NotImplementedError


# Memory Mechanism 을 위한 학습된 S-BERT (Sentence BERT) 모델을 이용하여 "각 example 에 대한" inference 실시
# Create Date : 2025.04.23
# Last Update Date : -

# Arguments:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델
# - memory_info (str)          : 메모리 정보 (예: "[오늘 일정: 친구와 저녁 식사]")
# - user_prompt (str)          : Oh-LoRA LLM 에 전달되는 사용자 프롬프트

# Returns:
# - similarity_score (float) : 학습된 S-BERT 모델이 계산한 similarity score (RAG 유사 메커니즘 용)

def run_inference_each_example(sbert_model, memory_info, user_prompt):
    raise NotImplementedError