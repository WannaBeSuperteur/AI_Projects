
## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. Oh-LoRA 👱‍♀️✨ (오로라) 소개](#1-1-oh-lora--오로라-소개)
  * [1-2. 실행 스크린샷](#1-2-실행-스크린샷)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 사용한 Python 라이브러리 및 시스템 환경](#2-1-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
  * [4-1. StyleGAN 을 이용한 이미지 생성](#4-1-stylegan-을-이용한-이미지-생성)
  * [4-2. LLM Fine-Tuning 을 이용한 사용자 대화 구현](#4-2-llm-fine-tuning-을-이용한-사용자-대화-구현)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. 사용자 가이드](#6-사용자-가이드)

## 1. 프로젝트 개요

**1. 핵심 아이디어**

* **LLM Fine-Tuning & StyleGAN** 을 이용한 가상인간 여성 [Oh-LoRA (오로라)](../2025_04_08_OhLoRA) 의 2차 업그레이드 버전
  * [1차 업그레이드 (Oh-LoRA v2)](../2025_05_02_OhLoRA_v2) 

**2. Oh-LoRA 👱‍♀️ (오로라) 이미지 생성 기술**

* [Oh-LoRA v2](../2025_05_02_OhLoRA_v2) 와 같이, 핵심 속성 값 (눈을 뜬 정도, 입을 벌린 정도, 고개 돌림 정도) 을 조정하는 벡터를 찾는 방법 사용
  * [참고 논문 스터디 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md) 

**3. LLM 관련 기술**

* 총 4 개의 LLM 에 대해 [Supervised Fine-Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md) 적용
  * 표정 생성을 위한 핵심 속성 값 역시, LLM 출력 답변 (예: ```눈: 크게 뜸```) 을 이용하여 결정
* 메모리 메커니즘
  * 현재 대화하고 있는 내용이 무엇인지를 파악, **화제 전환 등 특별한 이유가 없으면, 답변 생성 시 해당 정보를 활용**
  * 메모리 메커니즘을 위한 [S-BERT (Sentence BERT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) 의 학습 데이터 증량 및 품질 향상

### 1-1. Oh-LoRA 👱‍♀️✨ (오로라) 소개

* 성별 및 나이
  * 👱‍♀️ 여성
  * 2025년 기준 22 세 (2003년 10월 11일 생)
* MBTI
  * ENTJ 
* 학교
  * 🏫 알파고등학교 (2019.03 - 2022.02)
  * 🏰 샘올대학교 인공지능학과 (2022.03 - ) 3학년 재학 중
* 특수 능력
  * 오로라의 빛✨ 으로 우리 모두의 인생을 밝게 비춰 주는 마법 능력
  * 사람이 아닌 AI 가상 인간만이 가질 수 있음
* 기타 잡다한 TMI
  * 오로라 Fine-Tuning 에 사용한 데이터셋 (직접 제작) 을 보면 알 수 있어요!
  * [dataset v2](llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2.csv), [dataset v2.1](llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2_1.csv), [dataset v2.2](llm/fine_tuning_dataset/OhLoRA_fine_tuning_v2_2.csv)

<details><summary>(스포일러) 오로라👱‍♀️ 가 2003년 10월 11일 생인 이유 [ 펼치기 / 접기 ] </summary>

오로라를 개발한 [개발자 (wannabesuperteur)](https://github.com/WannaBeSuperteur) 가 개발할 때 Python 3.10.11 을 사용했기 때문이다.

</details>

### 1-2. 실행 스크린샷

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Image Generation (Generative AI)
  * LLM (Large Language Model)
* 사용 기술

| 기술 분야            | 사용 기술                                                                                                                                                                                         | 설명                                                                                                                                                                                                    |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Image Generation | StyleGAN **(+ Condition Vector Finding)**                                                                                                                                                     | 가상 인간 이미지 생성                                                                                                                                                                                          |
| Image Generation | [SVM (Support Vector Machine)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%AA%A8%EB%8D%B8_SVM.md) | 핵심 속성 값을 변화시키는 벡터를 탐색하기 위한 머신러닝 모델                                                                                                                                                                    |
| LLM              | [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)                                     | 가상 인간이 인물 설정에 맞게 사용자와 대화할 수 있게 하는 기술                                                                                                                                                                  |
| LLM              | [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md)                                  | 가상 인간의 LLM 을 효율적으로 Fine-Tuning 하는 기술                                                                                                                                                                  |
| LLM              | [S-BERT (Sentence BERT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8)      | 가상 인간이 사용자와의 대화 내용을 기억하는 메모리 역할<br>- [RAG (Retrieval Augmented Generation)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md) 과 유사한 메커니즘 |
| LLM              | [BERT](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#bert-%EB%AA%A8%EB%8D%B8%EC%9D%B4%EB%9E%80)       | 가상 인간이 사용자의 질문이 **화제 전환인지 / 부적절한 언어를 사용했는지** 판단                                                                                                                                                       |

### 2-1. 사용한 Python 라이브러리 및 시스템 환경

* Python
  * Python : **Python 3.10.11**
  * Dev Tool : PyCharm 2024.1 Community Edition
* Python Libraries
  * 주요 파이썬 라이브러리 (TBU)
  * 실험 환경의 전체 파이썬 라이브러리 목록 (TBU)
* OS & CPU & GPU
  * OS : **Windows 10**
  * CPU : Intel(R) Xeon(R) CPU E5-2690 0 @ 2.90GHz
  * GPU : 2 x **Quadro M6000** (12 GB each)
  * **CUDA 12.4** (NVIDIA-SMI 551.61)
* 시스템 환경 상세 정보 (TBU)

## 3. 프로젝트 일정

* 전체 일정 : **2025.05.26 월 - 06.03 화 (9d)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❎ (DONE BUT **NOT MERGED**), ❌ (FAILED)

**1. 프로젝트 전체 관리**

| 구분       | 계획 내용                      | 일정                     | branch                 | issue | 상태 |
|----------|----------------------------|------------------------|------------------------|-------|----|
| 📃 문서화   | 프로젝트 개요 및 최초 일정 작성         | 05.26 월 (1d)           |                        |       | ✅  |
| 🔍 최종 검토 | 최종 사용자 실행용 코드 작성           | 06.02 월 (1d)           | ```P005-011-ForUser``` |       | ⬜  |
| 🔍 최종 검토 | 최종 QA (버그 유무 검사)           | 06.02 월 - 06.03 화 (2d) |                        |       | ⬜  |
| 📃 문서화   | 데이터셋 및 모델 HuggingFace 에 등록 | 06.03 화 (1d)           |                        |       | ⬜  |
| 📃 문서화   | 프로젝트 문서 정리 및 마무리           | 06.03 화 (1d)           |                        |       | ⬜  |

**2. StyleGAN 을 이용한 가상 인간 얼굴 생성**

| 구분       | 계획 내용                                                                                                                                             | 일정                     | branch                                | issue                                                              | 상태 |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------|--------------------------------------------------------------------|----|
| 🛠 사전 작업 | [StyleGAN-FineTune-v1](../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan) 구현 및 사람 얼굴 이미지 1만 장 생성       | 05.26 월 (1d)           | ```P005-001-StyleGAN-FineTune-v1```   | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/14) | 💨 |
| 🛠 사전 작업 | 성별, 이미지 품질, 나이, 안경 여부 예측 CNN 학습을 위한 라벨링 (각 2,000 장)                                                                                               | 05.26 월 (1d)           | ```P005-002-CNN```                    |                                                                    | ⬜  |
| 🧪 모델 학습 | 성별, 이미지 품질, 나이, 안경 여부 예측 CNN 학습                                                                                                                   | 05.27 화 (1d)           | ```P005-002-CNN```                    |                                                                    | ⬜  |
| 🛠 사전 작업 | 성별, 이미지 품질, 나이, 안경 여부 조건에 따라, 생성된 얼굴 이미지 필터링                                                                                                      | 05.27 화 (1d)           | ```P005-002-CNN```                    |                                                                    | ⬜  |
| 🛠 사전 작업 | '곱슬머리 or 직모' 핵심 속성 값 라벨링 알고리즘 개발                                                                                                                  | 05.27 화 (1d)           | ```P005-003-Hairstyle```              |                                                                    | ⬜  |
| 🧪 모델 학습 | '곱슬머리 or 직모' 핵심 속성 값 예측 CNN 개발 및 학습                                                                                                               | 05.27 화 - 05.28 수 (2d) | ```P005-003-Hairstyle```              |                                                                    | ⬜  |
| 🛠 사전 작업 | '곱슬머리 or 직모' 핵심 속성 값 예측 CNN 을 기존 [Property Score CNN](../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-3-cnn-model-나머지-핵심-속성-값-7개) 과 결합 | 05.28 수 (1d)           | ```P005-003-Hairstyle```              |                                                                    | ⬜  |
| 🧪 모델 학습 | 필터링된 이미지를 StyleGAN-FineTune-v1 으로 추가 Fine-Tuning **(StyleGAN-FineTune-v8)**                                                                       | 05.28 수 - 05.29 목 (2d) | ```P005-004-StyleGAN-FineTune-v8```   |                                                                    | ⬜  |
| 🔨 모델 개선 | StyleGAN-FineTune-v8 에서 핵심 속성 값만 변화시키는 vector 추출 구현 시, '곱슬머리 or 직모' 핵심 속성 값 추가 **(StyleGAN-VectorFind-v8)**                                       | 05.29 목 (1d)           | ```P005-005-StyleGAN-VectorFind-v8``` |                                                                    | ⬜  |
| 🧪 모델 학습 | StyleGAN-FineTune-v8 에서 핵심 속성 값만 변화시키는 vector 추출 학습 및 성능 테스트                                                                                      | 05.29 목 - 05.30 금 (2d) | ```P005-005-StyleGAN-VectorFind-v8``` |                                                                    | ⬜  |
| 📃 문서화   | StyleGAN 개발 내용 문서화                                                                                                                                | 05.28 수 - 05.30 금 (3d) |                                       |                                                                    | ⬜  |

**3. LLM 을 이용한 대화 능력 향상**

| 구분         | 계획 내용                                                                                              | 일정                     | branch                     | issue | 상태 |
|------------|----------------------------------------------------------------------------------------------------|------------------------|----------------------------|-------|----|
| 🧪 모델 학습   | [Kanana-1.5-2.1B (by Kakao)](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-base) Fine-Tuning 시도 | 05.30 금 (1d)           | ```P005-006-Kanana```      |       | ⬜  |
| 📝 데이터셋 작성 | LLM Supervised Fine-Tuning 학습 데이터 증량                                                               | 05.30 금 - 05.31 토 (2d) |                            |       | ⬜  |
| 🔨 모델 개선   | LLM Supervised Fine-Tuning 학습 데이터 Augmentation 시도                                                  | 05.31 토 (1d)           | ```P005-007-LLM-Augment``` |       | ⬜  |
| 🧪 모델 학습   | LLM Supervised Fine-Tuning 학습 **(1차, 모델 4개)**                                                      | 05.31 토 - 06.01 일 (2d) | ```P005-008-train-LLM```   |       | ⬜  |
| 📝 데이터셋 작성 | BERT (화제 전환 여부 파악) 학습 데이터 작성                                                                       | 06.01 일 (1d)           | ```P005-009-BERT-topic```  |       | ⬜  |
| 🧪 모델 개선   | BERT (화제 전환 여부 파악) 학습 (Pre-trained BERT 기반)                                                        | 06.01 일 (1d)           | ```P005-009-BERT-topic```  |       | ⬜  |
| 📝 데이터셋 작성 | BERT (부적절한 언어 여부 파악) 학습 데이터 작성                                                                     | 06.02 월 (1d)           | ```P005-010-BERT-ethics``` |       | ⬜  |
| 🧪 모델 학습   | BERT (부적절한 언어 여부 파악) 학습 (Pre-trained BERT 기반)                                                      | 06.02 월 (1d)           | ```P005-010-BERT-ethics``` |       | ⬜  |
| 📃 문서화     | LLM 개발 내용 문서화                                                                                      | 06.02 월 (1d)           |                            |       | ⬜  |

## 4. 프로젝트 상세 설명

* 사용자의 질문에 대해 **가상 인간 여성 Oh-LoRA 👱‍♀️ (오로라)** 가 답변 생성
  * 이때 주변 환경 및 사용자에 대한 정보 (예: ```[오늘 날씨: 맑음]``` ```[내일 일정: 친구랑 카페 방문]```) 를 Oh-LoRA 의 메모리에 저장
  * Oh-LoRA 는 메모리에 있는 내용 중 가장 관련된 내용을 참고하여 답변
  * 가장 관련된 내용이 없을 경우, **직전 대화 turn 의 요약된 내용을 기억** 하고, **화제 전환 등이 아닌 경우** 그 요약 정보를 참고하여 답변
* Oh-LoRA 의 답변 내용에 따라 가상 인간 여성 이미지 생성
  * **"눈을 뜬 정도, 입을 벌린 정도, 고개 돌림" 의 3가지 속성 값** 을, LLM 의 답변에 기반하여 LLM 으로 생성한 표정 정보 (자연어) 에 따라 적절히 결정

### 4-1. StyleGAN 을 이용한 이미지 생성

* StyleGAN 의 핵심 속성 값을 변화시키는 벡터를 찾고, 해당 벡터를 이용하는 방법 적용
* [참고 논문 스터디 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)

### 4-2. LLM Fine-Tuning 을 이용한 사용자 대화 구현

* (TBU) 모델을 600 rows 규모의 학습 데이터셋으로 Fine-Tuning
* [RAG (Retrieval Augmented Generation)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md) 과 유사한 컨셉으로 LLM 의 memory 구현

## 5. 프로젝트 진행 중 이슈 및 해결 방법

## 6. 사용자 가이드

