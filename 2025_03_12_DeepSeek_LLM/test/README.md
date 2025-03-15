## 목차

* [1. 테스트 목적](#1-테스트-목적)
* [2. 테스트 환경](#2-테스트-환경)
* [3. 테스트 진행 및 결과](#3-테스트-진행-및-결과)
  * [3-1. 코드 파일 설명 및 테스트 프롬프트](#3-1-코드-파일-설명-및-테스트-프롬프트)
  * [3-2. 후보 모델 선정](#3-2-후보-모델-선정)
  * [3-3. Quantization 방법](#3-3-quantization-방법)
  * [3-4. 각 LLM 별 상세 결과](#3-4-각-llm-별-상세-결과)

## 1. 테스트 목적

* 본 프로젝트에서 어떤 LLM 을 Fine-tuning 하여 사용하는 것이 가장 좋은지를 확인한다.
* 이를 위해, HuggingFace 에서 사용 가능한 DeepSeek LLM 중 **일정한 기준에 따라 선정한 후보 모델 (14개)** 들을 한눈에 비교 분석할 수 있는 report 를 출력한다.
* 이 report 를 통해 비교 분석하여 최종 결론을 도출한다.

## 2. 테스트 환경

* 선택한 환경
  * Google Colab
  * Auto-GPTQ 가 지원되지 않는 큰 모델
    * T4 GPU (15GB) + 고용량 RAM → **T4 GPU (15GB) + 일반 RAM**
  * Auto-GPTQ 가 지원되는 작은 모델
    * **A100 GPU (40GB) + 일반 RAM** 
* 이유
  * 다음 로컬 환경에서 대부분의 LLM이 Out-of-memory 발생
    * OS : **Windows 10**
    * Deep Learning Framework : **PyTorch 2.4.0+cu124**
    * CUDA : **Version 12.4**
    * GPU : **Quadro M6000 (12 GB)**
  * T4 GPU (15GB) 만으로도, Auto-GPTQ 가 지원되지 않는 큰 모델들을 제외하고 모두 테스트 가능
  * A100 GPU (40GB) 수준의 대용량 GPU를 사용해야 해당 큰 모델들에서 Out-of-memory 가 발생하지 않음

## 3. 테스트 진행 및 결과

* 최종 Fine-Tuning 할 모델 (1차 테스트)
  * 14개 LLM 중, **deepseek-coder-1.3b-instruct** 를 채택
* 이유
  * 테스트 프롬프트를 이용하여 생성한 답변에 대한 Human Evaluation 결과, **DeepSeek-Coder-V2-Lite-Instruct** 과 함께 최고 품질 판정
  * 최고 품질의 답변을 생성한 2개의 모델 중 **deepseek-coder-1.3b-instruct** 이 메모리 사용량 및 응답 시간 측면에서 훨씬 우수함
    * Fine-Tuning 도 비교적 빨리 진행할 수 있을 것으로 기대됨 

| 구분     | 설명                                             | 결과                                            | 채택 모델                            |
|--------|------------------------------------------------|-----------------------------------------------|----------------------------------|
| 1차 테스트 | 테스트 프롬프트로 **모든 후보 모델로 1번씩 생성**, 최선의 모델 탐색      | 규모가 비교적 작은 8개의 모델 중에서도 고품질 답변이 나오는 모델이 있음을 확인 | **deepseek-coder-1.3b-instruct** |
| 2차 테스트 | 변경된 프롬프트로 **비교적 작은 8개 모델로 20번씩 생성**, 최선의 모델 탐색 | **(최종)**                                      | **(최종)**                         |

### 3-1. 코드 파일 설명 및 테스트 프롬프트

* 코드 파일
  * ```create_llm_report.py```
  * 1차 및 2차 테스트에서 공통으로 해당 코드 사용 (단, 일부 변경 사항 있음)
* 입/출력
  * 입력 : 없음
  * 출력 : ```llm_report.csv``` (LLM 이름, 정상 작동 여부, 각 LLM 의 사용 메모리, 응답 시간, [양자화](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md) 필요 여부, 테스트 프롬프트에 대한 출력값을 정리한 report)
* 테스트 프롬프트 (1차 테스트)

```
Represent below as a Python list.

A deep learning model with 2 input nodes, 4 and 6 nodes in each of the 2 hidden layers,
and 1 node in the output layer in the following format.

At this time, each node is represented in the format of "[node No., X position, Y position, shape,
connection line shape, background color, connection line color,
list of node No. s of other nodes pointed to by the connection line]".

At this time, the color is represented in the format of tuple (R, G, B), in range 0-255, and
X position range is 0-1000 and Y position range is 0-600.
```

* 테스트 프롬프트 (2차 테스트)
  * 1차 테스트 프롬프트에서 ```width``` ```height``` 도형 속성 추가 및 ```(px)``` 를 추가 명시

```
Represent below as a Python list.

A deep learning model with 2 input nodes, 4 and 6 nodes in each of the 2 hidden layers,
and 1 node in the output layer in the following format.

At this time, each node is represented in the format of Python list "[node No., X position (px), Y position (px), shape,
width (px), height (px), connection line shape, background color, connection line color,
list of node No. s of other nodes pointed to by the connection line]".

At this time, the color is represented in the format of tuple (R, G, B), in range 0-255, and
X position range is 0-1000 and Y position range is 0-600.
```

* 테스트 프롬프트에서 다음 부분만이 실제 User Prompt 이고, 나머지는 [프롬프트 엔지니어링](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Prompt_Engineering.md) 을 위해 추가한 부분임.

```
A deep learning model with 2 input nodes, 4 and 6 nodes in each of the 2 hidden layers,
and 1 node in the output layer in the following format.
```
  
### 3-2. 후보 모델 선정

* [HuggingFace DeepSeek Repo.](https://huggingface.co/deepseek-ai) 에서 다음을 모두 만족시키는 모델만 후보로 선정
  * Text-to-Text LLM
  * 모델 파라미터 개수는 20B 미만 (응답 시간 및 메모리 사용량 최소화)
  * 코딩을 제외한 특정 분야 (수학 등) 에 최적화되지는 않은 모델
  * DeepSeek-R1 또는 이를 [Distillation](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Knowledge_Distillation.md) 한 모델 등 [추론형 모델](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_%EC%B6%94%EB%A1%A0%ED%98%95_%EB%AA%A8%EB%8D%B8.md) 은 후보에서 제외 (추론으로 인해 응답 시간이 긺)
* 후보 선정 결과
  * 총 14 개 모델
    * 1차 테스트 : 모든 모델 대상
    * 2차 테스트 : DeepSeek-V2, DeepSeek-MoE 계열을 제외한 8개 모델 대상
  * DeepSeek-V2
    * [```DeepSeek-V2-Lite``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) (Auto-GPTQ Not Supported)
    * [```DeepSeek-V2-Lite-Chat``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) (Auto-GPTQ Not Supported)
    * [```DeepSeek-Coder-V2-Lite-Base``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base) (Auto-GPTQ Not Supported)
    * [```DeepSeek-Coder-V2-Lite-Instruct``` (15.7B)](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) (Auto-GPTQ Not Supported)
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
    * [```deepseek-moe-16b-chat``` (16.4B)](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) (Auto-GPTQ Not Supported)
    * [```deepseek-moe-16b-base``` (16.4B)](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base) (Auto-GPTQ Not Supported)
* 참고
  * DeepSeek-MoE 는 **Mixture of Expert** 라는 기법을 적용한 LLM 임 
  * **MoE (Mixture of Expert, 전문가 혼합)**
    * 사용자 입력 프롬프트에 따라 서로 다른 네트워크 (전문가) 를 활성화
    * 이를 통해, 파라미터는 많아지지만 **연산량은 유지** 

### 3-3. Quantization 방법

Quantization 방법은 [GPTQ](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md#2-4-gptq-post-training-quantization-for-gpt-models) 를 사용

* 본 프로젝트 특성상 **개발 일정에 맞춘 빠르고 효율적인 양자화**가 필요
* [PTQ (Post-training Quantization)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md#2-1-ptq-vs-qat) 방법론 적용
  * 이미 Pre-train 된 LLM 의 빠른 Fine-tuning 에는 QAT 보다 PTQ 가 적합
* PTQ 방법론 중, **3,4비트로 양자화해도 손실이 적으며 학습 효율성도 높은** GPTQ 방법론 사용

### 3-4. 각 LLM 별 상세 결과

* 모든 모델은 양자화 적용 없이 적절한 환경에서 실행 성공함

| 모델                                            | 사용 메모리    | 답변 시간  | 답변 품질 (1차 테스트)         | 답변 품질 (2차 테스트)     | 테스트 환경   |
|-----------------------------------------------|-----------|--------|------------------------|--------------------|----------|
| DeepSeek-V2-Lite                              | 31,126 MB | 92.1 s | 기본 형식 미준수              |                    | A100 GPU |
| DeepSeek-V2-Lite-Chat                         | 31,126 MB | 63.1 s | 기본 형식 부분적 준수           |                    | A100 GPU |
| **DeepSeek-Coder-V2-Lite-Base**               | 31,148 MB | 90.7 s | **기본 형식 준수**           |                    | A100 GPU |
| **DeepSeek-Coder-V2-Lite-Instruct**           | 31,148 MB | 66.8 s | **기본 형식 준수 + 비교적 고품질** |                    | A100 GPU |
| deepseek-coder-6.7b-instruct                  | 12,857 MB | 26.3 s | 기본 형식 미준수              | 형식 준수 **N / 20 개** | T4 GPU   |
| **deepseek-coder-7b-instruct-v1.5**           | 13,180 MB | 25.3 s | **기본 형식 준수**           | 형식 준수 **N / 20 개** | T4 GPU   |
| **deepseek-coder-1.3b-instruct<br>(✅ 최종 채택)** | 2,576 MB  | 19.5 s | **기본 형식 준수 + 비교적 고품질** | 형식 준수 **N / 20 개** | T4 GPU   |
| deepseek-coder-6.7b-base                      | 12,865 MB | 24.8 s | 기본 형식 부분적 준수           | 형식 준수 **N / 20 개** | T4 GPU   |
| deepseek-coder-7b-base-v1.5                   | 13,188 MB | 26.1 s | 기본 형식 미준수              | 형식 준수 **N / 20 개** | T4 GPU   |
| deepseek-coder-1.3b-base                      | 2,576 MB  | 22.9 s | 기본 형식 미준수              | 형식 준수 **N / 20 개** | T4 GPU   |
| **deepseek-llm-7b-chat**                      | 13,189 MB | 31.9 s | **기본 형식 준수**           | 형식 준수 **N / 20 개** | T4 GPU   |
| deepseek-llm-7b-base                          | 13,189 MB | 26.8 s | 기본 형식 미준수              | 형식 준수 **N / 20 개** | T4 GPU   |
| deepseek-moe-16b-chat                         | 31,475 MB | 53.5 s | 기본 형식 부분적 준수           |                    | A100 GPU |
| deepseek-moe-16b-base                         | 31,472 MB | 86.2 s | 기본 형식 미준수              |                    | A100 GPU |

**1. DeepSeek-Coder-V2-Lite-Base 의 답변**

* 모양 및 색과 관련된 모든 항목에 대해서, 그 값이 모든 node 에서 일정함

```python
[
    [0, 100, 100, "circle", "solid", (0, 0, 0), (0, 0, 0), [1, 2]],
    [1, 300, 100, "circle", "solid", (0, 0, 0), (0, 0, 0), [3, 4]],
    [2, 500, 100, "circle", "solid", (0, 0, 0), (0, 0, 0), [3, 4]],
    [3, 100, 300, "circle", "solid", (0, 0, 0), (0, 0, 0), [5]],
    [4, 300, 300, "circle", "solid", (0, 0, 0), (0, 0, 0), [5]],
    [5, 500, 300, "circle", "solid", (0, 0, 0), (0, 0, 0), [6]],
    [6, 500, 500, "circle", "solid", (0, 0, 0), (0, 0, 0), []]
]
```

**2. DeepSeek-Coder-V2-Lite-Instruct 의 답변**

* 대체로 의도에 맞는 적절한 답변으로 보임

```python
model = [
    [1, 100, 100, 'ellipse', 'vee', '#FF0000', '#00FF00', [2, 3]],
    [2, 200, 200, 'ellipse', 'vee', '#00FF00', '#FF0000', [4]],
    [3, 300, 300, 'ellipse', 'vee', '#FF0000', '#00FF00', [4, 5]],
    [4, 400, 100, 'ellipse', 'vee', '#00FF00', '#FF0000', [6]],
    [5, 500, 200, 'ellipse', 'vee', '#FF0000', '#00FF00', [6]],
    [6, 600, 300, 'ellipse', 'vee', '#00FF00', '#FF0000', []]
]
```

**3. deepseek-coder-7b-instruct-v1.5 의 답변**

* 형식을 준수한 다른 답변들과 달리, 앞에 line number 가 표시됨
* 생성 token 개수 제한으로 인해 나머지 부분이 잘려 나간 것으로 추정됨 (token 개수 제한만 해결하면 됨)

```python
1: [1, 100, 100, 'circle', 'solid', (255, 0, 0), [2, 3]]
2: [2, 200, 200, 'rectangle', 'dashed', (0, 255, 0), [3]]
3: [3, 300, 300, 'triangle', 'dotted', (0, 0, 255), [4, 5]]
4: [4, 400, 400, 'pentagon', 'dash_dot', (255, 255, 0), [5]]
5: [5, 500, 500, 'hexagon', 'solid', (0, 255, 255), [
```

**4. deepseek-coder-1.3b-instruct 의 답변**

* 도형의 색과 모양을 최대한 다양하게 하여 생성한 것으로 보임
* 생성 token 개수 제한으로 인해 나머지 부분이 잘려 나간 것으로 추정됨 (token 개수 제한만 해결하면 됨)

```python
model = [
    [1, 500, 300, "circle", "dashed", (255, 0, 0), (0, 0, 0), [2, 3]],
    [2, 700, 300, "square", "dashed", (0, 255, 0), (0, 0, 0), [3, 4]],
    [3, 500, 500, "triangle", "dashed", (0, 0, 255), (0, 0, 0), [1, 2]],
    [4, 700, 500, "rectangle", "dashed", (0, 255, 255), (0, 0, 0), [2, 3]],
    [5, 500, 700, "pentagon", "dashed", (255, 255, 0), (0, 0, 0), [4, 5]],
    [6, 700, 700, "hexagon", "dashed", (255, 0, 255), (0, 0, 0), [5, 6]],
    [7, 500, 900, "octagon", "dashed", (0, 255, 255), (0, 0, 0), [6, 7]],
```

**5. deepseek-llm-7b-chat 의 답변**

* 형식은 준수했으나, 구현 의도와는 거리가 다소 있는 것으로 보임

```python
model = [
    [
        [2, 100, 200, "circle", "none", (255, 255, 255), (255, 255, 255), []],
        [4, 300, 300, "square", "none", (0, 0, 0), (255, 255, 255), [1, 2, 3]],
        [6, 400, 400, "square", "none", (0, 0, 0), (255, 255, 255), [1, 3]],
    ],
    [
        [8, 450, 450, "square", "none", (0, 0, 0), (255, 255, 255), [4, 6]],
        [10, 500, 500, "circle", "none", (0, 0, 0), (255, 255, 255), [4]],
    ],
]
```