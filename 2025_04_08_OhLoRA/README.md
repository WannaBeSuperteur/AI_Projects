## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. 프로젝트 진행 배경](#1-1-프로젝트-진행-배경)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 관련 논문](#2-1-관련-논문)
  * [2-2. 사용한 Python 라이브러리 및 시스템 환경](#2-2-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
  * [4-1. StyleGAN 을 이용한 이미지 생성](#4-1-stylegan-을-이용한-이미지-생성)
  * [4-2. 이미지 평가 및 추천](#4-2-이미지-평가-및-추천)
  * [4-3. LLM Fine-Tuning 을 이용한 사용자 대화 구현](#4-3-llm-fine-tuning-을-이용한-사용자-대화-구현)
  * [4-4. RAG 을 이용한 메모리 구현](#4-4-rag-을-이용한-메모리-구현) 
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. 사용자 가이드](#6-사용자-가이드)
* [7. 프로젝트 소감](#7-프로젝트-소감)

## 1. 프로젝트 개요

**1. 핵심 아이디어**

* StyleGAN 및 LLM 을 응용한, 사용자와 대화하는 가상 인간 여성 캐릭터 **(이름 : Oh-LoRA (오로라))** 생성

**2. 주요 내용 (이미지 생성 및 추천)**

* Fine-Tuning 된 StyleGAN 을 이용하여 이미지 생성
  * 이때, 다음과 같은 **핵심 속성** 값을 이용하여 이미지 생성
  * 핵심 속성 값 (7가지)
    * 성별, 이미지 품질, 눈을 뜬 정도, 머리 색, 머리 길이, 입을 벌린 정도, 고개 돌림, 배경색 평균, 배경색 표준편차
    * 성별, 이미지 품질이 **모두 조건을 충족시키는 이미지만 따로 필터링** 하여, 필터링된 이미지에 대해서만 나머지 7가지 속성 값 적용
    * 성별, 이미지 품질을 제외한 값은 **Pre-trained Segmentation Model 을 이용하여 라벨링**
  * **핵심 속성** 값에 해당하는 벡터를 StyleGAN 의 latent vector 에 추가
    * 해당 부분만 학습 가능하게 하고, **나머지 파라미터는 모두 Freeze 처리** 

* N 장의 생성된 이미지마다 **예상 사용자 평가 점수** 를 매겨서, 그 값이 높은 이미지들을 추천
  * 예상 사용자 평가 점수
    * Auto-Encoder 로 이미지를 저차원 벡터로 압축 후,
    * 학습 데이터 (= 생성된 이미지에 대한 사용자 평가 데이터) 에 있는 이웃한 저차원 벡터들에 대응되는 이미지들에 대한 **평가 점수의 가중 평균** 으로 산출
    * 이때 [k-NN (k-Nearest Neighbors)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%AA%A8%EB%8D%B8_KNN.md) 를 이용

**3. 주요 내용 (LLM 을 이용한 대화)**

* LLM 을 Fine-Tuning (LoRA 이용) 하여, **가상 인간 설정에 맞게** 사용자와 대화
* [RAG (Retrieval Augmented Generation)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md) 을 이용하여, 향후에도 기억해야 할 중요한 내용을 메모리에 저장
  * RAG 에 저장할 중요한 정보도 LLM 을 이용하여 파악
  * RAG 을 통해 사용자에 대한 경고/차단 시스템도 구현
* LLM 의 답변에 대해, Oh-LoRA 캐릭터의 **핵심 속성 값을 그 답변에 맞게 적절히 변경 (예: 놀라는 말투 답변의 경우 → 눈을 크게 뜸) 하여 이미지를 생성** 하는 메커니즘 구현

**4. 이름 Oh-LoRA (오로라) 의 의미**

* 내 인생은 오로라처럼 밝게 빛날 것이라는 자기 확신 (개발자 본인 & 사용자 모두에게)
* [LLM Fine-Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning.md) 방법 중 최근 널리 쓰이는 [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md) 에서 유래

### 1-1. 프로젝트 진행 배경

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Computer Vision
  * Image Generation (Generative AI)
  * LLM (Large Language Model)
* 사용 기술

| 기술 분야            | 사용 기술                                                                                                                                                     | 설명                                                                                                                                                                                                                                                                              |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Image Generation | StyleGAN                                                                                                                                                  | 가상 인간 이미지 생성                                                                                                                                                                                                                                                                    |
| Computer Vision  | Segmentation                                                                                                                                              | StyleGAN 으로 생성한 가상 인간 이미지의 **핵심 속성** 값 추출                                                                                                                                                                                                                                       |
| Computer Vision  | CNN (Conv. NN)                                                                                                                                            | StyleGAN 으로 생성한 가상 인간 이미지 중 성별 값이 있는 2,000 장을 학습 후, 학습된 CNN 으로 제외한 나머지 이미지들의 **성별 값** 추론                                                                                                                                                                                        |
| Computer Vision  | Auto-Encoder                                                                                                                                              | 생성된 이미지의 저차원 벡터화를 통해, k-NN 을 통한 사용자 평가 예상 점수 계산 시 **이웃한 이미지와의 거리 계산이 정확해지고, 연산량이 감소하는** 효과                                                                                                                                                                                      |
| LLM              | [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md) | 가상 인간이 인물 설정에 맞게 사용자와 대화할 수 있게 하는 기술                                                                                                                                                                                                                                            |
| LLM              | [RAG (Retrieval Augmented Generation)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md)     | 가상 인간이 사용자와의 대화 내용을 기억하는 메모리 역할                                                                                                                                                                                                                                                 |
| LLM              | [KTO](https://arxiv.org/pdf/2402.01306) (Optional)                                                                                                        | 가상 인간의 대화 성능 추가 향상 (참고 : [LLM Survey 논문 Study 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Large%20Language%20Model/%5B2025.03.22%5D%20Large%20Language%20Models%20A%20Survey%20(IV%2C%20VII).md#1-1-g-llm-%EC%97%90-%EB%8C%80%ED%95%9C-alignment)) |
|                  | k-NN                                                                                                                                                      | StyleGAN 으로 생성한 가상 인간 이미지의 평가 데이터 (점수) 에 기반한, **평가 예상 점수 계산** 알고리즘                                                                                                                                                                                                              |

### 2-1. 관련 논문

본 프로젝트에서 사용할 기술에 대한 **탄탄한 기초가 중요하다** 는 판단 아래 작성한, 관련 논문에 관한 스터디 자료이다.

* **StyleGAN** (이미지 생성 기술) 논문
  * [A Style-Based Generator Architecture for Generative Adversarial Networks (2018.12)](https://arxiv.org/pdf/1812.04948)
* **KTO** (LLM Fine-Tuning 기술) 논문
  * [KTO: Model Alignment as Prospect Theoretic Optimization (2024.02)](https://arxiv.org/pdf/2402.01306)

### 2-2. 사용한 Python 라이브러리 및 시스템 환경

## 3. 프로젝트 일정

* 전체 일정 : **2025.04.08 화 - 04.17 목 (10d)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❌ (FAILED)

**프로젝트 전체 관리**

| 계획 내용                                              | 일정                     | branch                         | 상태 |
|----------------------------------------------------|------------------------|--------------------------------|----|
| 프로젝트 개요 작성                                         | 04.08 화 (1d)           |                                | ✅  |
| Python 3.10 으로 업그레이드                               | 04.09 수 (1d)           |                                | ✅  |
| 최종 테스트 및 QA                                        | 04.16 수 (1d)           |                                | ⬜  |
| 프로젝트 마무리 및 문서 정리                                   | 04.17 목 (1d)           |                                | ⬜  |

**이미지 생성 (StyleGAN)**

| 계획 내용                                              | 일정                     | branch                         | 상태 |
|----------------------------------------------------|------------------------|--------------------------------|----|
| 논문 탐독 (StyleGAN)                                   | 04.09 수 (1d)           |                                | ✅  |
| 기본 StyleGAN 구현                                     | 04.09 수 - 04.10 목 (2d) | ```P002-001-StyleGAN```        | ✅  |
| 이미지 품질 판단 (CNN 구현 포함)                              | 04.10 목 - 04.11 금 (2d) | ```P002-002-CNN```             | ✅  |
| 성별 속성 값 추출 (CNN 구현 포함)                             | 04.10 목 - 04.11 금 (2d) | ```P002-002-CNN```             | ✅  |
| Segmentation 모델 구현 (핵심 속성 값 추출용)                   | 04.10 목 - 04.11 금 (2d) | ```P002-003-Seg```             | ✅  |
| Segmentation 으로 핵심 속성 값 추출                         | 04.11 금 (1d)           | ```P002-003-Seg```             | ✅  |
| 핵심 속성 값을 처리하도록 StyleGAN 구조 변경 (Fine-Tuning 포함, 1차) | 04.11 금 - 04.12 토 (2d) | ```P002-004-Update-StyleGAN``` | ✅  |
| **(추가)** 핵심 속성 값 추가 (배경 색 밝기, 배경 색 표준편차)           | 04.12 토 (1d)           | ```P002-005-Property```        | ✅  |
| 추가된 핵심 속성 값으로 StyleGAN 학습 (Fine-Tuning, 2차)        | 04.12 토 - 04.13 일 (2d) | ```P002-006-Update-StyleGAN``` | 💨 |
| Auto-Encoder 구현                                    | 04.13 일 (1d)           | ```P002-007-AE```              | ⬜  |
| k-NN 구현                                            | 04.13 일 (1d)           | ```P002-008-kNN```             | ⬜  |

**거대 언어 모델 (LLM)**

| 계획 내용                                              | 일정                     | branch                         | 상태 |
|----------------------------------------------------|------------------------|--------------------------------|----|
| 논문 탐독 (KTO)                                        | 04.09 수 (1d)           |                                | ✅  |
| Unsloth 실행 시도                                      | 04.14 월 (1d)           |                                | ⬜  |
| 적절한 한국어 LLM 모델 선택                                  | 04.14 월 (1d)           |                                | ⬜  |
| SFT 학습 데이터셋 제작                                     | 04.14 월 (1d)           | ```P002-009-SFT-Dataset```     | ⬜  |
| SFT + LoRA 를 이용한 Fine-Tuning                       | 04.14 월 - 04.15 화 (2d) | ```P002-010-SFT-LoRA```        | ⬜  |
| KTO 를 이용한 추가 Fine-Tuning                           | 04.15 화 (1d)           | ```P002-011-KTO```             | ⬜  |
| RAG 적용                                             | 04.15 화 - 04.16 수 (2d) | ```P002-012-RAG```             | ⬜  |

## 4. 프로젝트 상세 설명

### 4-1. StyleGAN 을 이용한 이미지 생성

### 4-2. 이미지 평가 및 추천

### 4-3. LLM Fine-Tuning 을 이용한 사용자 대화 구현

### 4-4. RAG 을 이용한 메모리 구현

## 5. 프로젝트 진행 중 이슈 및 해결 방법

* [해당 문서](issue_reported.md) 참고.

## 6. 사용자 가이드

## 7. 프로젝트 소감