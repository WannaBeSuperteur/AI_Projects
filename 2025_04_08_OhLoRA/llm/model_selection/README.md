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
    * **1.5B parameter 이하** (Quadro M6000 12GB 에서 학습 및 추론 가능한 수준이어야 함)
    * 자유 사용 가능 라이선스 (MIT, Apache-2.0 등)
    * 한국어 특화 또는 지원
  * 위 기준에 **셋 다** 해당하는 모델 중 **종합 점수 (Average)** 가 높은 순으로 테스트
* 모델 선정 기준
  * 아래 기준을 모두 고려하여 **최종 1개 모델 선정**
    * Open Ko-LLM Leaderboard 기준 종합 점수 **(정량 평가)** 
    * Fine-Tuning 속도 및 Fine-Tuning 진행 중 최대 메모리 사용량 **(정량 평가)**
    * Inference 속도 및 Inference 진행 중 최대 메모리 사용량 **(정량 평가)**
    * [Fine-Tuning 학습 데이터셋](../OhLoRA_fine_tuning.csv) 의 Valid Data 에 대한 답변 품질 **(정성 평가)**
  * 기준 데이터셋
    * [Fine-Tuning 학습 데이터셋](../OhLoRA_fine_tuning.csv) 의 Train Data 및 Valid Data 의 일부

## 3. 모델 선정 결과

| 모델 후보 선정 결과                     | 최종 선정 모델 정보 |
|---------------------------------|-------------|
| 총 6개 모델 (0.04 B ~ 1.5 B params) |             |

### 3-1. 모델 후보 선정 결과

| 순위 | LLM 이름                                                                              | 파라미터 개수 | 종합 점수 (Average) |
|----|-------------------------------------------------------------------------------------|---------|-----------------|
| 1  | [koquality-polyglot-1.3b](https://huggingface.co/DILAB-HYU/koquality-polyglot-1.3b) | 1.3 B   | 31.07           |
| 2  | [polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)              | 1.44 B  | 29.95           |
| 3  | [llama1B](https://huggingface.co/Yebin46/llama1B)                                   | 1.24 B  | 29.73           |
| 4  | [KoAlpaca-KoRWKV-1.5B](https://huggingface.co/beomi/KoAlpaca-KoRWKV-1.5B)           | 1.5 B   | 28.74           |
| 5  | [FinguAI-Chat-v1](https://huggingface.co/FINGU-AI/FinguAI-Chat-v1)                  | 0.46 B  | 28.09           |
| 6  | [TinyKo-v5-a](https://huggingface.co/blueapple8259/TinyKo-v5-a)                     | 0.04 B  | 25.80           |

### 3-2. 최종 모델 선정 결과

TBU