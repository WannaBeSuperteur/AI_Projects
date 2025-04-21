# LLM Model Selection for Oh-LoRA Fine-Tuning

## 목차

* [1. 개요](#1-개요)
* [2. 모델 선정 기준](#2-모델-선정-기준)
* [3. 모델 선정 결과](#3-모델-선정-결과)
  * [3-1. 모델 후보 선정 결과](#3-1-모델-후보-선정-결과)
  * [3-2. 최종 모델 선정 결과](#3-2-최종-모델-선정-결과)

## 1. 개요

Oh-LoRA (오로라) 의 [Fine-Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning.md) 을 위한 적절한 LLM 을 선택한다.

## 2. 모델 선정 기준

* 모델 후보 선정 시 참고 페이지
  * [HuggingFace 의 Open Ko-LLM Leaderboard (2025.04.19 기준)](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)
* 모델 후보 선정 기준
  * 선정 기준 
    * **0.25 B - 3.0 B 사이의 parameter** (Quadro M6000 12GB 에서 학습 및 추론 가능한 수준이어야 함)
    * 자유 사용 가능 라이선스 (MIT, Apache-2.0 등)
    * 한국어 특화 또는 지원
  * 위 기준에 **셋 다** 해당하는 모델 중 **종합 점수 (Average)** 가 높은 순으로 테스트
* 모델 선정 기준
  * 아래 기준을 모두 고려하여 **최종 1개 모델 선정**
    * Open Ko-LLM Leaderboard 기준 종합 점수 **(정량 평가)** 
    * Fine-Tuning 속도 및 Fine-Tuning 진행 중 최대 메모리 사용량 **(정량 평가)**
      * LoRA Rank 16, 32, 64 로 각각 테스트 
    * Inference 속도 및 Inference 진행 중 최대 메모리 사용량 **(정량 평가)**
      * Inference test 는 **Fine-Tuning 되지 않은 원본 모델** 로 진행 
    * [Fine-Tuning 학습 데이터셋](../OhLoRA_fine_tuning.csv) 에 대한 답변 품질 **(정성 평가)**
      * 단, LLM Fine Tuning 테스트 중 기록되는 Training Loss 값을 참고하여 평가 
  * 기준 데이터셋
    * [Fine-Tuning 학습 데이터셋](../OhLoRA_fine_tuning.csv) 의 Train Data 및 Valid Data 의 일부

## 3. 모델 선정 결과

| 모델 후보 선정 결과                       | 최종 선정 모델                                                                                        |
|-----------------------------------|-------------------------------------------------------------------------------------------------|
| 총 7 개 모델 (0.46 B ~ 2.61 B params) | **gemma-2-2b-it (2.61 B params)** [(HuggingFace)](https://huggingface.co/unsloth/gemma-2-2b-it) |

### 3-1. 모델 후보 선정 결과

* 총 7 개 모델 후보 선정
* 파라미터 개수 범위 : 0.46 B ~ 2.61 B

| 순위 | LLM 이름                                                                              | 파라미터 개수 | 종합 점수 (Average) |
|----|-------------------------------------------------------------------------------------|---------|-----------------|
| 1  | [gemma-2-2b-it](https://huggingface.co/unsloth/gemma-2-2b-it)                       | 2.61 B  | 40.43           |
| 2  | [koquality-polyglot-1.3b](https://huggingface.co/DILAB-HYU/koquality-polyglot-1.3b) | 1.3 B   | 31.07           |
| 3  | [gemma-ko-v01](https://huggingface.co/cpm-ai/gemma-ko-v01)                          | 2.51 B  | 30.78           |
| 4  | [polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)              | 1.43 B  | 29.95           |
| 5  | [llama1B](https://huggingface.co/Yebin46/llama1B)                                   | 1.24 B  | 29.73           |
| 6  | [KoAlpaca-KoRWKV-1.5B](https://huggingface.co/beomi/KoAlpaca-KoRWKV-1.5B)           | 1.5 B   | 28.74           |
| 7  | [FinguAI-Chat-v1](https://huggingface.co/FINGU-AI/FinguAI-Chat-v1)                  | 0.46 B  | 28.09           |

### 3-2. 최종 모델 선정 결과

**1. 최종 모델 선정 결과**

* **gemma-2-2b-it (2.61 B params)**
  * [License : gemma (NOT totally free)](https://ai.google.dev/gemma/terms) 
* **리포트 (ipynb) 는 GitHub 에서 읽히지 않으므로, 다운로드 후 Google Colab 또는 PyCharm 에서 읽는 것을 권장** 합니다.

| 구분           | 모델 이름 (리포트)                                                    | 종합 점수<br>(Average) | 파라미터 개수 | Fine-Tuning 시간 (LoRA rank 16 / 32 / 64) | Fine-Tuning 최대 메모리 사용량 (LoRA rank 16 / 32 / 64) | Inference 시간 | Inference 최대 메모리 사용량 | Inference 답변 품질                                     |
|--------------|----------------------------------------------------------------|--------------------|---------|-----------------------------------------|-------------------------------------------------|--------------|----------------------|-----------------------------------------------------|
| **최종 선정 모델** | gemma-2-2b-it [(ipynb)](test_result_gemma_2_2b_it.ipynb)       | **40.43**          | 2.61 B  | 58.7 s / 63.0 s / 67.0 s                | 5,249 MB / 5,553 MB / 5,969 MB                  | 4.03 s       | 4,987 MB             | [**최상**](inference_test_result_gemma-2-2b-it.csv)   |
| 예비 모델        | polyglot-ko-1.3b [(ipynb)](test_result_polyglot-ko-1.3b.ipynb) | 29.95              | 1.43 B  | 24.2 s / 24.8 s / 26.2 s                | 2,593 MB / 2,629 MB / 2,715 MB                  | 1.78 s       | 2,541 MB             | [**상**](inference_test_result_polyglot-ko-1.3b.csv) |

**2. 모델 후보 평가 실험 설정**

* 환경
  * Google Colab, T4 GPU (15 GB RAM) 
* Fine-Tuning 설정
  * data (row) count = **20**, train batch size = **4**, epochs = **10**
  * total **50 steps** with **size-4 batch** for each step **(= 실제 학습 데이터셋에서의 1 epoch 분량)**
* Inference 설정
  * data (row) count = **25**, max output length (tokens) = **80** 

**3. 모델 후보 평가 결과**

* **리포트 (ipynb) 는 GitHub 에서 읽히지 않으므로, 다운로드 후 Google Colab 또는 PyCharm 에서 읽는 것을 권장** 합니다.

| 모델 이름 (리포트)                                                            | 종합 점수<br>(Average) | 파라미터 개수    | Fine-Tuning 시간 (LoRA rank 16 / 32 / 64) | Fine-Tuning 최대 메모리 사용량 (LoRA rank 16 / 32 / 64) | Inference 시간 (평균) | Inference 메모리 사용량 (최대) | Inference 답변 품질                                        |
|------------------------------------------------------------------------|--------------------|------------|-----------------------------------------|-------------------------------------------------|-------------------|------------------------|--------------------------------------------------------|
| gemma-2-2b-it [(ipynb)](test_result_gemma_2_2b_it.ipynb)               | **40.43**          | 2.61 B     | 58.7 s / 63.0 s / 67.0 s                | 5,249 MB / 5,553 MB / 5,969 MB                  | 4.03 s            | 4,987 MB               | [**최상**](inference_test_result_gemma-2-2b-it.csv)      |
| koquality-polyglot-1.3b [(ipynb)](test_result_polyglot-ko-1.3b.ipynb)  | 31.07              | 1.3 B      | 23.7 s / 24.5 s / 28.4 s                | 2,593 MB / 2,629 MB / 2,715 MB                  | 6.68 s            | 2,541 MB               | [중](inference_test_result_koquality-polyglot-1.3b.csv) |
| gemma-ko-v01 [(ipynb)](test_result_gemma_ko_v01.ipynb)                 | 30.78              | 2.51 B     | 55.9 s / 61.5 s / 64.7 s                | 5,021 MB / 5,246 MB / 5,694 MB                  | 2.06 s            | 4,780 MB               | [중](inference_test_result_gemma-ko-v01.csv)            |
| polyglot-ko-1.3b [(ipynb)](test_result_polyglot-ko-1.3b.ipynb)         | 29.95              | 1.43 B     | 24.2 s / 24.8 s / 26.2 s                | 2,593 MB / 2,629 MB / 2,715 MB                  | 1.78 s            | 2,541 MB               | [**상**](inference_test_result_polyglot-ko-1.3b.csv)    |
| llama1B [(ipynb)](test_result_llama1B.ipynb)                           | 29.73              | 1.24 B     | 26.2 s / 26.8 s / 28.7 s                | 2,503 MB / 2,632 MB / 2,894 MB                  | 1.54 s            | 2,358 MB               | [중](inference_test_result_llama1B.csv)                 |
| KoAlpaca-KoRWKV-1.5B [(ipynb)](test_result_KoAlpaca-KoRWKV-1.5B.ipynb) | 28.74              | 1.5 B      | 80.0 s / 76.0 s / 79.8 s                | 3,083 MB / 3,245 MB / 3,572 MB                  | 2.08 s            | 2,905 MB               | [하](inference_test_result_KoAlpaca-KoRWKV-1.5B.csv)    |
| FinguAI-Chat-v1 [(ipynb)](test_result_FinguAI-Chat-v1.ipynb)           | 28.09              | **0.46 B** | **15.8 s / 16.2 s / 17.0 s**            | **1,009 MB / 1,096 MB / 1,269 MB**              | **0.46 s**        | **906 MB**             | [최하](inference_test_result_FinguAI-Chat-v1.csv)        |