## 목차

* [1. 테스트 환경](#1-테스트-환경)
* [2. LLM 에 대한 비교 Report 생성](#2-llm-에-대한-비교-report-생성)
  * [2-1. 코드 파일 설명](#2-1-코드-파일-설명)
  * [2-2. 후보 모델 선정](#2-2-후보-모델-선정)
  * [2-3. 실제 Report 결과](#2-3-실제-report-결과)

## 1. 테스트 환경

* OS : **Windows 10**
* Deep Learning Framework : **PyTorch 2.4.0+cu124**
* CUDA : **Version 12.4**
* GPU : ```Quadro M6000``` 

## 2. LLM 에 대한 비교 Report 생성

본 프로젝트에서 어떤 LLM 을 사용하는 것이 가장 좋은지를 알기 위해, 사용 가능한 DeepSeek LLM 중 **일정한 기준에 따라 선정한 후보 모델**들을 한눈에 비교 분석할 수 있는 report 를 출력한다.

### 2-1. 코드 파일 설명

* 코드 파일
  * ```create_llm_report.py```
* 입/출력
  * 입력 : 없음
  * 출력 : ```llm_report.csv``` (LLM 이름, 성공 여부, 각 LLM 의 사용 메모리, 응답 시간, [양자화](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md) 필요 여부, 테스트 프롬프트에 대한 출력값을 정리한 report)
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
  * ```DeepSeek-V2-Lite``` (15.7B)
  * ```DeepSeek-V2-Lite-Chat``` (15.7B)
  * ```DeepSeek-Coder-V2-Lite-Base``` (15.7B)
  * ```DeepSeek-Coder-V2-Lite-Instruct``` (15.7B)
  * ```deepseek-coder-6.7b-instruct``` (6.74B)
  * ```deepseek-coder-7b-instruct-v1.5``` (6.91B)
  * ```deepseek-coder-1.3b-instruct``` (1.35B)
  * ```deepseek-coder-6.7b-base``` (6.74B)
  * ```deepseek-coder-7b-base-v1.5``` (6.91B)
  * ```deepseek-coder-1.3b-base``` (1.35B)
  * ```deepseek-llm-7b-chat``` (7B)
  * ```deepseek-llm-7b-base``` (7B)
  * ```deepseek-moe-16b-base``` (16.4B)
  * ```deepseek-moe-16b-chat``` (16.4B)

### 2-3. 실제 Report 결과

TBU