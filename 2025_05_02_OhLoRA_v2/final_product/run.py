import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import sys
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from stylegan.stylegan_common.stylegan_generator import StyleGANGeneratorForV6
from llm.memory_mechanism.load_sbert_model import load_pretrained_sbert_model


# 필요한 모델 로딩 : StyleGAN-VectorFind-v7 Generator, 4 LLMs (Polyglot-Ko 1.3B Fine-Tuned), S-BERT (RoBERTa-based)
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - stylegan_generator    (nn.Module)       : StyleGAN-VectorFind-v7 generator
# - ohlora_llms           (dict(LLM))       : LLM (Polyglot-Ko 1.3B Fine-Tuned)
#                                             {'output_message': LLM, 'memory': LLM, 'summary': LLM,
#                                              'eyes_mouth_pose': LLM}
# - ohlora_llms_tokenizer (dict(tokenizer)) : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
#                                             {'output_message': tokenizer, 'memory': tokenizer, 'summary': tokenizer,
#                                              'eyes_mouth_pose': tokenizer}
# - sbert_model           (S-BERT Model)    : S-BERT (RoBERTa-based)

def load_models():
    gpu_0 = torch.device('cuda:0')
    gpu_1 = torch.device('cuda:1')

    output_types = ['output_message', 'memory', 'eyes_mouth_pose', 'summary']
    device_mapping = {'output_message': gpu_0, 'memory': gpu_0, 'eyes_mouth_pose': gpu_1, 'summary': gpu_1}

    # load StyleGAN-VectorFind-v7 generator model
    stylegan_generator = StyleGANGeneratorForV6(resolution=256)  # v6 and v7 have same architecture
    stylegan_model_dir = f'{PROJECT_DIR_PATH}/stylegan/models'
    generator_path = f'{stylegan_model_dir}/stylegan_gen_vector_find_v7.pth'

    generator_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
    stylegan_generator.load_state_dict(generator_state_dict)
    stylegan_generator.to(device)

    # load Oh-LoRA final LLM and tokenizer
    ohlora_llms = {}
    ohlora_llms_tokenizer = {}

    for output_type in output_types:
        model_path = f'{PROJECT_DIR_PATH}/llm/models/polyglot_{output_type}_fine_tuned'
        ohlora_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).to(device_mapping[output_type])

        ohlora_llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        ohlora_llm.generation_config.pad_token_id = ohlora_llm_tokenizer.pad_token_id

        ohlora_llms[output_type] = ohlora_llm
        ohlora_llms_tokenizer[output_type] = ohlora_llm_tokenizer

    # load S-BERT Model (RoBERTa-based)
    model_path = f'{PROJECT_DIR_PATH}/llm/models/memory_sbert/trained_sbert_model'
    sbert_model = load_pretrained_sbert_model(model_path)

    return stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model


# Oh-LoRA (오로라) 실행
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments:
# - stylegan_generator    (nn.Module)       : StyleGAN-VectorFind-v7 generator
# - ohlora_llms           (dict(LLM))       : LLM (Polyglot-Ko 1.3B Fine-Tuned)
#                                             {'output_message': LLM, 'memory': LLM, 'summary': LLM,
#                                              'eyes_mouth_pose': LLM}
# - ohlora_llms_tokenizer (dict(tokenizer)) : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
#                                             {'output_message': tokenizer, 'memory': tokenizer, 'summary': tokenizer,
#                                              'eyes_mouth_pose': tokenizer}
# - sbert_model           (S-BERT Model)    : S-BERT (RoBERTa-based)

# Running Mechanism:
# - Oh-LoRA LLM 답변 생성 시마다 이에 기반하여 final_product/ohlora.png 경로에 오로라 이미지 생성
# - Oh-LoRA 답변을 parsing 하여 llm/memory_mechanism/saved_memory/ohlora_memory.txt 경로에 메모리 저장
# - S-BERT 모델을 이용하여, RAG 와 유사한 방식으로 해당 파일에서 사용자 프롬프트에 가장 적합한 메모리 정보를 찾아서 최종 LLM 입력에 추가

def run_ohlora(stylegan_generator, ohlora_llm, ohlora_llm_tokenizer, sbert_model):
    raise NotImplementedError


if __name__ == '__main__':

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # load model
    stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model = load_models()
    print('ALL MODELS for Oh-LoRA (오로라) load successful!! 👱‍♀️')

    # run Oh-LoRA (오로라)
    run_ohlora(stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model)
