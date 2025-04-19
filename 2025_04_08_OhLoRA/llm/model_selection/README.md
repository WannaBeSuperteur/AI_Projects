# LLM Model Selection for Oh-LoRA Fine-Tuning

## 목차

* [1. 개요](#1-개요)
* [2. 모델 선정 기준](#2-모델-선정-기준)
* [3. 모델 선정 결과](#3-모델-선정-결과)

## 1. 개요

Oh-LoRA (오로라) 의 [Fine-Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning.md) 을 위한 적절한 LLM 을 선택한다.

## 2. 모델 선정 기준

* 모델 후보 선정 시 참고 페이지
  * [HuggingFace 의 Open Ko-LLM Leaderboard (2025.04.19 기준)](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)
* 모델 후보 선정 기준
  * **1.5B parameter 이하** (Quadro M6000 12GB 에서 학습 및 추론 가능한 수준이어야 함)
  * 위 기준에 해당하는 모델 중 **종합 점수 (Average)** 가 높은 순으로 테스트
* 모델 선정 기준
  * 아래 기준에 따라 **최종 1개 모델 선정** 
    * Fine-Tuning 진행 중 최대 메모리 사용량 **(정량 평가)**
    * Inference 진행 중 최대 메모리 사용량 **(정량 평가)**
    * [Fine-Tuning 학습 데이터셋](../OhLoRA_fine_tuning.csv) 의 Valid Data 에 대한 답변 품질 **(정성 평가)**

## 3. 모델 선정 결과

TBU