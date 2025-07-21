## 목차

* [1. LLM Fine-Tuning (해설용)](#1-llm-fine-tuning-해설용)
* [2. S-BERT Training (사용자 답변 채점용)](#2-s-bert-training-사용자-답변-채점용)
* [3. 틀릴 가능성이 높은 퀴즈 출제](#3-틀릴-가능성이-높은-퀴즈-출제)
* [4. 코드 실행 방법](#4-코드-실행-방법)
  * [4-1. LLM Fine-Tuning (해설용)](#4-1-llm-fine-tuning-해설용)
  * [4-2. S-BERT Training (사용자 답변 채점용)](#4-2-s-bert-training-사용자-답변-채점용)
  * [4-3. 틀릴 가능성이 높은 퀴즈 출제](#4-3-틀릴-가능성이-높은-퀴즈-출제)

## 1. LLM Fine-Tuning (해설용)

## 2. S-BERT Training (사용자 답변 채점용)

## 3. 틀릴 가능성이 높은 퀴즈 출제

## 4. 코드 실행 방법

모든 코드는 **먼저 LLM 모델 정보 및 다운로드 경로 안내 (TBU) 및 해당 각 HuggingFace 링크에 있는 Model Card 에 나타난 저장 경로 (Save Path) 정보를 참고하여 모델 다운로드 후,** ```2025_06_24_OhLoRA_v4``` (프로젝트 메인 디렉토리) 에서 실행

### 4-1. LLM Fine-Tuning (해설용)

지정된 경로에 해당 LLM 이 이미 존재하는 경우, Fine-Tuning 대신 **inference test 실행됨**

* ```python ai_quiz/run_llm_fine_tuning.py``` (for [Kanana-1.5 2.1B Instruct](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505))
* ```python ai_quiz/run_llm_fine_tuning.py -llm_name midm``` (for [Mi:dm 2.0 Mini](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct))

### 4-2. S-BERT Training (사용자 답변 채점용)

지정된 경로에 해당 S-BERT 모델이 이미 존재하는 경우, Fine-Tuning 대신 **inference test 실행됨**

```python ai_quiz/run_sbert.py```

### 4-3. 틀릴 가능성이 높은 퀴즈 출제

[Quiz Selection 알고리즘 (IoU Score 기반 가중치에 따른 사용자 점수 예측 결과)](#3-틀릴-가능성이-높은-퀴즈-출제) 에 따라, **사용자 예측 점수가 가장 낮은** 퀴즈 출제

```python ai_quiz/run_select_quiz.py```