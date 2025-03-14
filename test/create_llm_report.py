TEST_PROMPT = ("Draw a diagram of neural network model with 4 input nodes, 6 first hidden layer nodes, 6 second "
               "hidden layer nodes, and 1 output node. For each node, write in the form of [node No., background color,"
               "text, text color, edge color, arrow color, connected nodes No. (as an array)]. Color should be such"
               "as #008000 format.")

MODEL_NAMES = ['DeepSeek-V2-Lite', 'DeepSeek-V2-Lite-Chat',
               'DeepSeek-Coder-V2-Lite-Base', 'DeepSeek-Coder-V2-Lite-Instruct',
               'deepseek-coder-6.7b-instruct', 'deepseek-coder-7b-instruct-v1.5', 'deepseek-coder-1.3b-instruct',
               'deepseek-coder-6.7b-base', 'deepseek-coder-7b-base-v1.5', 'deepseek-coder-1.3b-base',
               'deepseek-llm-7b-chat', 'deepseek-llm-7b-base',
               'deepseek-moe-16b-chat', 'deepseek-moe-16b-base']




# 각 LLM을 테스트하고, [LLM 이름, 성공 여부, 사용 메모리, 응답(추론) 시간, 양자화 필요 여부, 테스트 프롬프트 출력값] 형식으로 반환
# Create Date : 2025.03.14
# Last Update Date : -

# Arguments:
# - model_name (str) : 모델 이름 (예: deepseek-llm-7b-chat)

# Returns:
# - 아래의 내용을 dict 로 묶어서 반환
#   - success     (bool)  : 성공 여부
#   - used_memory (float) : 사용 메모리 양 (MB)
#   - resp_time   (float) : 응답(추론) 시간 (초)
#   - quant_need  (bool)  : 양자화 (Quantization) 필요 여부
#   - test_output (str)   : 테스트 프롬프트에 대한 출력값

def test_llm(model_name):
    return NotImplementedError


# 테스트 성공한 LLM에 한해, 그 LLM을 파일로 저장
# Create Date : 2025.03.14
# Last Update Date : -

# Arguments:
# - model_name (str) : 모델 이름 (예: deepseek-llm-7b-chat)

# Outputs:
# - models/{model_name} 경로에 해당 모델 파일 저장

def save_llm(model_name):
    return NotImplementedError


# 테스트 성공하여 저장한 LLM을 로딩
# Create Date : 2025.03.14
# Last Update Date : -

# Arguments:
# - model_name (str) : 모델 이름 (예: deepseek-llm-7b-chat)

# Returns:
# - llm : 로딩된 LLM

def load_llm(model_name):
    return NotImplementedError


def main():
    return NotImplementedError


if __name__ == '__main__':
    main()
