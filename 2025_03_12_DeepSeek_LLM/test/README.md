## 목차

* [1. 테스트 환경](#1-테스트-환경)
* [2. LLM 에 대한 비교 Report 생성](#2-llm-에-대한-비교-report-생성)
  * [2-1. 코드 파일 설명](#2-1-코드-파일-설명)
  * [2-2. 후보 모델 선정](#2-2-후보-모델-선정)
  * [2-3. Quantization 방법](#2-3-quantization-방법)
  * [2-4. 실제 Report 결과](#2-4-실제-report-결과)

## 1. 테스트 환경

* 선택한 환경
  * Google Colab
  * T4 GPU + 고용량 RAM
* 이유
  * 다음 로컬 환경에서 대부분의 LLM이 Out-of-memory 발생
    * OS : **Windows 10**
    * Deep Learning Framework : **PyTorch 2.4.0+cu124**
    * CUDA : **Version 12.4**
    * GPU : ```Quadro M6000``` (12 GB)

## 2. LLM 에 대한 비교 Report 생성

본 프로젝트에서 어떤 LLM 을 사용하는 것이 가장 좋은지를 알기 위해, 사용 가능한 DeepSeek LLM 중 **일정한 기준에 따라 선정한 후보 모델**들을 한눈에 비교 분석할 수 있는 report 를 출력한다.

### 2-1. 코드 파일 설명

* 코드 파일
  * ```create_llm_report.py```
* 입/출력
  * 입력 : 없음
  * 출력 : ```llm_report.csv``` (LLM 이름, 정상 작동 여부, 각 LLM 의 사용 메모리, 응답 시간, [양자화](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md) 필요 여부, 테스트 프롬프트에 대한 출력값을 정리한 report)
* 테스트 프롬프트
  * ```Draw a diagram of neural network model with 4 input nodes, 6 first hidden layer nodes, 6 second hidden layer nodes, and 1 output node. For each node, write in the form of [node No., background color, text, text color, edge color, arrow color, connected nodes No. (as an array)]. Color should be such as #008000 format.```

### 2-2. 후보 모델 선정

* [HuggingFace DeepSeek Repo.](https://huggingface.co/deepseek-ai) 에서 다음을 모두 만족시키는 모델만 후보로 선정
  * Text-to-Text LLM
  * 모델 파라미터 개수는 20B 미만 (응답 시간 및 메모리 사용량 최소화)
  * 코딩을 제외한 특정 분야 (수학 등) 에 최적화되지는 않은 모델
  * DeepSeek-R1 또는 이를 [Distillation](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Knowledge_Distillation.md) 한 모델 등 [추론형 모델](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_%EC%B6%94%EB%A1%A0%ED%98%95_%EB%AA%A8%EB%8D%B8.md) 은 후보에서 제외 (추론으로 인해 응답 시간이 긺)
* 후보 선정 결과
  * 총 14 개 모델
  * DeepSeek-V2
    * [```DeepSeek-V2-Lite``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
    * [```DeepSeek-V2-Lite-Chat``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)
    * [```DeepSeek-Coder-V2-Lite-Base``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base)
    * [```DeepSeek-Coder-V2-Lite-Instruct``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)
  * DeepSeek-Coder
    * [```deepseek-coder-6.7b-instruct``` (6.74B)](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
    * [```deepseek-coder-7b-instruct-v1.5``` (6.91B)](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
    * [```deepseek-coder-1.3b-instruct``` (1.35B)](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
    * [```deepseek-coder-6.7b-base``` (6.74B)](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base)
    * [```deepseek-coder-7b-base-v1.5``` (6.91B)](https://huggingface.co/deepseek-ai/deepseek-coder-7b-base-v1.5)
    * [```deepseek-coder-1.3b-base``` (1.35B)](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
  * DeepSeek-LLM
    * [```deepseek-llm-7b-chat``` (7B)](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
    * [```deepseek-llm-7b-base``` (7B)](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)
  * DeepSeek-MoE
    * [```deepseek-moe-16b-chat``` (16.4B)](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)
    * [```deepseek-moe-16b-base``` (16.4B)](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base)
* 참고
  * DeepSeek-MoE 는 **Mixture of Expert** 라는 기법을 적용한 LLM 임 
  * **MoE (Mixture of Expert, 전문가 혼합)**
    * 사용자 입력 프롬프트에 따라 서로 다른 네트워크 (전문가) 를 활성화
    * 이를 통해, 파라미터는 많아지지만 **연산량은 유지** 

### 2-3. Quantization 방법

Quantization 방법은 [GPTQ](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md#2-4-gptq-post-training-quantization-for-gpt-models) 를 사용

* 본 프로젝트 특성상 **개발 일정에 맞춘 빠르고 효율적인 양자화**가 필요
* [PTQ (Post-training Quantization)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md#2-1-ptq-vs-qat) 방법론 적용
  * 이미 Pre-train 된 LLM 의 빠른 Fine-tuning 에는 QAT 보다 PTQ 가 적합
* PTQ 방법론 중, **3,4비트로 양자화해도 손실이 적으며 학습 효율성도 높은** GPTQ 방법론 사용

### 2-4. 실제 Report 결과

TBU