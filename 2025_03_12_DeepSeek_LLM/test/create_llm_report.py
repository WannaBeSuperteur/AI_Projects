from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import time
import pandas as pd

quantize_config = BaseQuantizeConfig(bits=4, group_size=128)

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

llm_report = pd.DataFrame(columns=['success', 'used_memory', 'resp_time', 'quant_need', 'test_resp', 'error_msg'])

# check cuda is available
assert torch.cuda.is_available(), "CUDA MUST BE AVAILABLE"
print(f'cuda is available with device {torch.cuda.get_device_name()}')


# 각 LLM을 테스트하고, [LLM 이름, 정상 작동 여부, 사용 메모리, 응답(추론) 시간, 양자화 필요 여부, 테스트 프롬프트 출력값] 형식으로 반환
# Create Date : 2025.03.14
# Last Update Date : -

# Arguments:
# - model_name (str) : 모델 이름 (예: deepseek-llm-7b-chat)

# Returns:
# - llm         (LLM)  : 테스트 결과 정상 작동 (오류 없이 정상 작동) 시 해당 LLM 반환, 실패 시 None
# - result_dict (dict) : 아래의 내용을 dict 로 묶어서 반환
#   - success     (bool)  : 성공(정상 작동) 여부
#   - used_memory (float) : 사용 메모리 양 (MB)
#   - resp_time   (float) : 응답(추론) 시간 (초)
#   - quant_need  (bool)  : 양자화 (Quantization) 필요 여부
#   - test_resp   (str)   : 테스트 프롬프트에 대한 출력값
#   - error_msg   (str)   : 실패 (오류) 시 오류 메시지

def test_llm(model_name):

    try:
        # 양자화 없이 시도
        llm = AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{model_name}",
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16).cuda()
        result_dict = test_loaded_llm(llm, quantized=False)

    except:
        # 양자화 적용해서 시도 (.cuda() 사용 불가)
        try:
            llm = AutoGPTQForCausalLM.from_pretrained(f"deepseek-ai/{model_name}",
                                                      trust_remote_code=True,
                                                      quantize_config=quantize_config,
                                                      torch_dtype=torch.bfloat16)

            result_dict = test_loaded_llm(llm, quantized=True)
        except Exception as e:
            return None, {'success': False, 'error_msg': str(e)}

    return llm, result_dict


# 로딩 된 LLM을 테스트하여 [정상 작동 여부, 사용 메모리, 응답(추론) 시간, 양자화 필요 여부, 테스트 프롬프트 출력값] 형식으로 반환
# Create Date : 2025.03.14
# Last Update Date : -

# Arguments:
# - llm       (LLM)  : 테스트 대상 LLM
# - quantized (bool) : 양자화 실시 여부

# Returns:
# - result_dict (dict) : 아래의 내용을 dict 로 묶어서 반환
#   - success     (bool)  : 성공(정상 작동) 여부
#   - used_memory (float) : 사용 메모리 양 (MB)
#   - resp_time   (float) : 응답(추론) 시간 (초)
#   - quant_need  (bool)  : 양자화 (Quantization) 필요 여부
#   - test_resp   (str)   : 테스트 프롬프트에 대한 출력값
#   - error_msg   (str)   : 실패 (오류) 시 오류 메시지

def test_loaded_llm(llm, quantized):
    model_name = llm.config.name_or_path
    used_memory = torch.cuda.memory_allocated()
    tokenizer = AutoTokenizer.from_pretrained(f"deepseek_model/{model_name}")

    start = time.time()
    inputs = tokenizer(TEST_PROMPT, return_tensors='pt').to(llm.device)
    outputs = llm.generate(**inputs, max_length=512)
    resp_time = time.time() - start
    test_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result_dict = {'success': False,
                   'used_memory': used_memory,
                   'resp_time': resp_time,
                   'quent_need': quantized,
                   'test_resp': test_resp}

    return result_dict


# 테스트 결과 정상 작동한 LLM에 한해, 그 LLM을 파일로 저장
# Create Date : 2025.03.14
# Last Update Date : -

# Arguments:
# - model_name (str) : 모델 이름 (예: deepseek-llm-7b-chat)
# - llm        (LLM) : 해당 이름으로 저장할 LLM

# Outputs:
# - models/{model_name} 경로에 해당 모델 파일 저장

def save_llm(llm, model_name):
    pass


# 테스트 결과 정상 작동하여 저장한 LLM을 로딩
# Create Date : 2025.03.14
# Last Update Date : -

# Arguments:
# - model_name (str) : 모델 이름 (예: deepseek-llm-7b-chat)

# Returns:
# - llm (LLM) : 로딩된 LLM

def load_llm(model_name):
    return NotImplementedError


def main():
    global llm_report

    for model_name in MODEL_NAMES:
        llm, result_dict = test_llm(model_name)

        result_dict_df = pd.DataFrame([result_dict])
        llm_report = pd.concat([llm_report, result_dict_df], ignore_index=True)
        llm_report.to_csv('llm_report.csv')
        print(f'{model_name} test finished with result: {result_dict}')

        if result_dict.get('success'):
            save_llm(llm, model_name)


if __name__ == '__main__':
    main()
