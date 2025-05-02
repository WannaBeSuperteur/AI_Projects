
## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 관련 논문](#2-1-관련-논문)
  * [2-2. 사용한 Python 라이브러리 및 시스템 환경](#2-2-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. 사용자 가이드](#6-사용자-가이드)

## 1. 프로젝트 개요

**1. 핵심 아이디어**

* **LLM Fine-Tuning & StyleGAN** 을 이용한 가상인간 여성 [Oh-LoRA (오로라)](../2025_04_08_OhLoRA) 의 1차 업그레이드 버전

**2. Oh-LoRA 👱‍♀️ (오로라) 이미지 생성 기술**

* Conditional GAN 이용
  * 기존의 [StyleGAN 의 Generator 를 Conditional VAE 의 Decoder 로 사용하는 방법](../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan) 은 한계가 있다고 판단

**3. LLM 관련 기술**

* [Supervised Fine-Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md) 학습 데이터 증량을 통한 성능 향상
  * 표정 생성을 위한 핵심 속성 값을 기존의 조건문 대신, LLM 출력 답변 (예: ```눈: 크게 뜸```) 을 이용하여 결정
* 메모리 메커니즘 강화
  * 현재 대화하고 있는 내용이 무엇인지를 파악
  * 메모리 메커니즘을 위한 [S-BERT (Sentence BERT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) 의 학습 데이터 증량 및 품질 향상

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Image Generation (Generative AI)
  * LLM (Large Language Model)
* 사용 기술

| 기술 분야            | 사용 기술                                                                                                                                                                                    | 설명                                                                                                                                                                                                    |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Image Generation | StyleGAN **(+ Conditional + Fine Tuning)**                                                                                                                                               | 가상 인간 이미지 생성                                                                                                                                                                                          |
| LLM              | [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)                                | 가상 인간이 인물 설정에 맞게 사용자와 대화할 수 있게 하는 기술                                                                                                                                                                  |
| LLM              | [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md)                             | 가상 인간의 LLM 을 효율적으로 Fine-Tuning 하는 기술                                                                                                                                                                  |
| LLM              | [S-BERT (Sentence BERT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) | 가상 인간이 사용자와의 대화 내용을 기억하는 메모리 역할<br>- [RAG (Retrieval Augmented Generation)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md) 과 유사한 메커니즘 |

### 2-1. 관련 논문

본 프로젝트에서 사용할 기술에 대한 **탄탄한 기초가 중요하다** 는 판단 아래 작성한, 관련 논문에 관한 스터디 자료이다.

* **Conditional StyleGAN** (이미지 생성 기술) 논문
  * [논문 : Art Creation with Multi-Conditional StyleGANs (2022.02)](https://arxiv.org/pdf/2202.11777)
  * 스터디 자료 (TBU)

### 2-2. 사용한 Python 라이브러리 및 시스템 환경

* Python
  * Python : **Python 3.10.11**
  * Dev Tool : PyCharm 2024.1 Community Edition
* Python Libraries
  * [주요 파이썬 라이브러리](system_info_and_user_guide.md#1-1-주요-python-라이브러리)
  * [실험 환경의 전체 파이썬 라이브러리 목록](system_info_and_user_guide.md#1-2-시스템에-설치된-전체-python-라이브러리)
* OS & GPU
  * OS : **Windows 10**
  * GPU : 2 x **Quadro M6000** (12 GB each)
  * **CUDA 12.4** (NVIDIA-SMI 551.61)

## 3. 프로젝트 일정

* 전체 일정 : **2025.05.02 금 - 05.16 금 (15d)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❌ (FAILED)

| 계획 내용                                                                                                                                              | 일정                     | branch                                    | issue | 상태 |
|----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-------------------------------------------|-------|----|
| 프로젝트 개요 및 일정 작성                                                                                                                                    | 05.02 금 (1d)           |                                           |       | ✅  |
| 논문 탐독 (Conditional StyleGAN)                                                                                                                       | 05.03 토 (1d)           |                                           |       | ⬜  |
| StyleGAN Discriminator 구조 변경 ([Property Score 도출용 CNN](../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-3-cnn-model-나머지-핵심-속성-값-7개) 구조로) | 05.03 토 (1d)           | ```P003-001-StyleGAN-Dis```               |       | ⬜  |
| StyleGAN 구조 추가 변경 (탐독한 논문 참고)                                                                                                                      | 05.04 일 (1d)           | ```P003-002-StyleGAN```                   |       | ⬜  |
| StyleGAN 학습 데이터 Augmentation                                                                                                                       | 05.04 일 - 05.05 월 (2d) | ```P003-003-Augment```                    |       | ⬜  |
| StyleGAN 학습                                                                                                                                        | 05.05 월 - 05.06 화 (2d) | ```P003-004-StyleGAN-Train```             |       | ⬜  |
| LLM Supervised Fine-Tuning 학습 데이터 증량                                                                                                               | 05.05 월 (1d)           |                                           |       | ⬜  |
| [KoreamLM (by QuantumAI)](https://huggingface.co/quantumaikr/KoreanLM) Fine-Tuning 시도                                                              | 05.06 화 (1d)           | ```P003-005-KoreanLM```                   |       | ⬜  |
| S-BERT (for memory) 학습 데이터 증량 및 품질 향상                                                                                                              | 05.06 화 (1d)           |                                           |       | ⬜  |
| S-BERT (for memory) ground-truth similarity score 계산 고도화                                                                                           | 05.06 화 (1d)           | ```P003-006-SBERT-sim```                  |       | ⬜  |
| 하고 있는 대화 요약을 S-BERT (for memory) 와 연동                                                                                                              | 05.07 수 (1d)           | ```P003-007-SBERT-current-conversation``` |       | ⬜  |              
| S-BERT (for memory) 임베딩 방법 개선 연구                                                                                                                   | 05.07 수 - 05.08 목 (2d) | ```P003-008-SBERT-embedding```            |       | ⬜  |
| 답변 생성 시도 및 성공/실패 처리 메커니즘 개선                                                                                                                        | 05.08 목 (1d)           | ```P003-009-Answer```                     |       | ⬜  |
| 실시간 표정 변화 적용                                                                                                                                       | 05.08 목 (1d)           | ```P003-010-Realtime```                   |       | ⬜  |
| 최종 사용자 실행용 코드 작성                                                                                                                                   | 05.08 목 (1d)           | ```P003-011-ForUser```                    |       | ⬜  |
| 최종 QA (버그 유무 검사)                                                                                                                                   | 05.09 금 (1d)           |                                           |       | ⬜  |
| 프로젝트 문서 정리 및 마무리                                                                                                                                   | 05.09 금 (1d)           |                                           |       | ⬜  |
| buffer time **(이력서 업데이트, 과제 전형, 면접 등 취업 일정에 대비한 여유 기간)**                                                                                           | 05.10 토 - 05.16 금 (7d) |                                           |       |    |

## 4. 프로젝트 상세 설명

TBU

## 5. 프로젝트 진행 중 이슈 및 해결 방법

TBU

## 6. 사용자 가이드

TBU