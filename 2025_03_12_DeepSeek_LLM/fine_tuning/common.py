# ORPO 용 데이터셋 생성할 때, 각 LLM output 의 score 평가
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - input_data  (str) : 사용자 입력 프롬프트 + Prompt Engeineering 된 부분
# - output_data (str) : LLM 의 출력 답변

# Returns:
# - score (float) : input_data 에 대한 output_data 의 적절성 평가 score

def compute_output_score(input_data, output_data):
    raise NotImplementedError