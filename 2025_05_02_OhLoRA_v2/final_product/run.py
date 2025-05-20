import torch


# 필요한 모델 로딩 : StyleGAN-VectorFind-v7 Generator, 4 LLMs (Polyglot-Ko 1.3B Fine-Tuned), S-BERT (RoBERTa-based)
# Create Date : 2025.05.20
# Last Update Date : -

# Arguments:
# - device (device) : 모델들을 mapping 시킬 device (GPU 등)

# Returns:
# - stylegan_generator    (nn.Module)       : StyleGAN-VectorFind-v7 generator
# - ohlora_llms           (dict(LLM))       : LLM (Polyglot-Ko 1.3B Fine-Tuned)
#                                             {'output_message': LLM, 'memory': LLM, 'summary': LLM,
#                                              'eyes_mouth_pose': LLM}
# - ohlora_llms_tokenizer (dict(tokenizer)) : LLM (Polyglot-Ko 1.3B Fine-Tuned) 에 대한 tokenizer
#                                             {'output_message': tokenizer, 'memory': tokenizer, 'summary': tokenizer,
#                                              'eyes_mouth_pose': tokenizer}
# - sbert_model           (S-BERT Model)    : S-BERT (RoBERTa-based)

def load_models(device):
    raise NotImplementedError


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
    stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model = load_models(device)
    print('ALL MODELS for Oh-LoRA (오로라) load successful!! 👱‍♀️')

    # run Oh-LoRA (오로라)
    run_ohlora(stylegan_generator, ohlora_llms, ohlora_llms_tokenizer, sbert_model)
