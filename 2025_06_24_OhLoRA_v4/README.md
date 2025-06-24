
## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. Oh-LoRA 👱‍♀️✨ (오로라) 소개](#1-1-oh-lora--오로라-소개)
  * [1-2. 실행 스크린샷](#1-2-실행-스크린샷)
  * [1-3. Oh-LoRA 얼굴 변화 애니메이션](#1-3-oh-lora-얼굴-변화-애니메이션)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 사용한 Python 라이브러리 및 시스템 환경](#2-1-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
  * [4-1. Segmentation 을 이용한 옴브레 염색 구현](#4-1-segmentation-을-이용한-옴브레-염색-구현)
  * [4-2. LLM Fine-Tuning 을 이용한 사용자 대화 구현](#4-2-llm-fine-tuning-을-이용한-사용자-대화-구현)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. 사용자 가이드](#6-사용자-가이드)

## 1. 프로젝트 개요

**1. 핵심 아이디어**

* **LLM Fine-Tuning & StyleGAN** 을 이용한 가상인간 여성 [Oh-LoRA (오로라)](../2025_04_08_OhLoRA) 의 3차 업그레이드 버전
  * [1차 업그레이드 (Oh-LoRA v2)](../2025_05_02_OhLoRA_v2) 
  * [2차 업그레이드 (Oh-LoRA v3)](../2025_05_26_OhLoRA_v3) 
  * [Oh-LoRA 👱‍♀️ (오로라) 얼굴 생성 방법 추가 연구 (Oh-LoRA v3.1)](../2025_06_07_OhLoRA_v3_1) 

**2. Oh-LoRA 👱‍♀️ (오로라) 이미지 변형 (옴브레 염색) 기술**

* [Oh-LoRA v1](../2025_04_08_OhLoRA) 및 [Oh-LoRA v3](../2025_05_26_OhLoRA_v3) 에서 사용했던 Pre-trained Segmentation 모델인 [FaceXFormer](https://kartik-3004.github.io/facexformer/) 모델 사용
* **(Oh-LoRA 가상 얼굴, FaceXFormer hair 영역 추출 결과)** 쌍을 학습 데이터로 하는, **비교적 경량화된** Segmentation Model 개발 ([Knowledge Distillation](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Knowledge_Distillation.md))
* 해당 Segmentation Model 에 **hair 영역으로 인식된 부분** 에 대해 옴브레 염색 적용

**3. LLM 관련 기술**

* 아래 [Oh-LoRA v2](../2025_05_02_OhLoRA_v2), [Oh-LoRA v3](../2025_05_26_OhLoRA_v3) 구현 컨셉 기반
  * 총 4 개의 LLM 에 대해 [Supervised Fine-Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md) 적용
    * 표정 생성을 위한 핵심 속성 값 역시, LLM 출력 답변 (예: ```눈: 크게 뜸```) 을 이용하여 결정
  * 메모리 메커니즘
    * 현재 대화하고 있는 내용이 무엇인지를 파악, **화제 전환 등 특별한 이유가 없으면, 답변 생성 시 해당 정보를 활용**
    * 메모리 메커니즘을 위한 [S-BERT (Sentence BERT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) 의 학습 데이터 증량 및 품질 향상
* **Oh-LoRA v4 업그레이드 사항 : AI 윤리 강화**
  * 답변 생성용 LLM이 **부적절한 답변 생성 요청에 대해 응답을 거부** 하도록 Fine-Tuning 적용 
  * 부적절한 답변 생성 요청을 감지하여 경고 및 차단하는 기능 업그레이드 (**S-BERT** 모델 이용)
  * [Oh-LoRA v3 윤리성 테스트 결과](../2025_05_26_OhLoRA_v3/llm/ai_ethics_test_report.md)

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
  * Fine-Tuning 데이터셋 : (TBU)

<details><summary>(스포일러) 오로라👱‍♀️ 가 2003년 10월 11일 생인 이유 [ 펼치기 / 접기 ] </summary>

오로라를 개발한 [개발자 (wannabesuperteur)](https://github.com/WannaBeSuperteur) 가 개발할 때 Python 3.10.11 을 사용했기 때문이다.

</details>

### 1-2. 실행 스크린샷

TBU

### 1-3. Oh-LoRA 얼굴 변화 애니메이션

TBU

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Computer Vision (Segmentation + **Knowledge Distillation**)
  * LLM (Large Language Model)
* 사용 기술

| 기술 분야           | 사용 기술                                                                                                                                                                                    | 설명                                                                                                                                                                                                    |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Computer Vision | Segmentation                                                                                                                                                                             | StyleGAN 으로 생성한 가상 인간 이미지에서 **hair 영역 추출**                                                                                                                                                            |
| Computer Vision | Segmentation **(Knowledge Distillation → 경량화)**                                                                                                                                          | 경량화된 Segmentation Model 생성을 통해 **옴브레 염색 적용된 Oh-LoRA 👱‍♀️ (오로라) 얼굴 이미지를 보다 실시간에 가깝게 생성**                                                                                                              |
| LLM             | [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)                                | 가상 인간이 인물 설정에 맞게 사용자와 대화할 수 있게 하는 기술                                                                                                                                                                  |
| LLM             | [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md)                             | 가상 인간의 LLM 을 효율적으로 Fine-Tuning 하는 기술                                                                                                                                                                  |
| LLM             | [S-BERT (Sentence BERT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) | 가상 인간이 사용자와의 대화 내용을 기억하는 메모리 역할<br>- [RAG (Retrieval Augmented Generation)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md) 과 유사한 메커니즘 |
| LLM             | [S-BERT (Sentence BERT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) | 가상 인간이 사용자의 질문이 **부적절한 언어를 사용했는지** 판단                                                                                                                                                                 |

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

* 전체 일정 : **2025.06.24 화 - 06.29 일 (6 days)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❎ (DONE BUT **NOT MERGED**), ❌ (FAILED)

**1. 프로젝트 전체 관리**

| 구분       | 계획 내용                      | 일정           | branch                 | issue | 상태 |
|----------|----------------------------|--------------|------------------------|-------|----|
| 📃 문서화   | 프로젝트 개요 및 최초 일정 작성         | 06.24 화 (1d) |                        |       | ✅  |
| 🔍 최종 검토 | 최종 사용자 실행용 코드 작성           | 06.28 토 (1d) | ```P007-007-ForUser``` |       | ⬜  |
| 🔍 최종 검토 | 최종 QA (버그 유무 검사)           | 06.28 토 (1d) |                        |       | ⬜  |
| 📃 문서화   | 데이터셋 및 모델 HuggingFace 에 등록 | 06.29 일 (1d) |                        |       | ⬜  |
| 📃 문서화   | 프로젝트 문서 정리 및 마무리           | 06.29 일 (1d) |                        |       | ⬜  |

**2. Segmentation 모델 경량화 & hair 영역 추출**

| 구분       | 계획 내용                                                                                                                                                                                                                                                                                                                                                  | 일정                     | branch                             | issue | 상태 |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|------------------------------------|-------|----|
| 🛠 사전 작업 | [StyleGAN-VectorFind-v7](../2025_05_02_OhLoRA_v2/stylegan/README.md#3-3-stylegan-finetune-v1-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v7) 및 [StyleGAN-VectorFind-v8](../2025_05_26_OhLoRA_v3/stylegan/README.md#3-3-stylegan-finetune-v8-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v8) 구현 **(최종 Oh-LoRA 이미지 생성 부분만)** | 06.24 화 (1d)           | ```P007-001-StyleGAN-VectorFind``` |       | ⬜  |
| 🛠 사전 작업 | Segmentation 학습 **입력** 데이터 준비<br>([StyleGAN-FineTune-v8](../2025_05_26_OhLoRA_v3/stylegan/README.md#3-2-fine-tuned-stylegan-stylegan-finetune-v8) 학습에 사용했던 [고품질 여성 얼굴 이미지 4,930 장](../2025_05_26_OhLoRA_v3/stylegan/README.md#1-1-모델-구조))                                                                                                              | 06.24 화 (1d)           | ```P007-002-Prepare-Data```        |       | ⬜  |
| 🛠 사전 작업 | FaceXFormer 모델 추가 구현 **(Knowledge Distillation 으로 하려면, 각 픽셀 별 soft label 정보 필요)**                                                                                                                                                                                                                                                                      | 06.24 화 (1d)           | ```P007-002-Prepare-Data```        |       | ⬜  |
| 🛠 사전 작업 | FaceXFormer 모델을 이용하여, 학습 입력 데이터에 대한 출력값 도출 **(→ 학습 데이터 완성)**                                                                                                                                                                                                                                                                                           | 06.24 화 - 06.25 수 (2d) | ```P007-002-Prepare-Data```        |       | ⬜  |
| 🛠 사전 작업 | Segmentation 학습 데이터 필터링 (Segmentation 결과의 'noise'가 큰 데이터 제거)                                                                                                                                                                                                                                                                                           | 06.25 수 (1d)           | ```P007-002-Prepare-Data```        |       | ⬜  |
| 🔨 모델 구현 | 경량화된 Segmentation 모델 구현                                                                                                                                                                                                                                                                                                                                | 06.25 수 (1d)           | ```P007-003-Segmentation```        |       | ⬜  |
| 🧪 모델 학습 | 경량화된 Segmentation 모델 학습                                                                                                                                                                                                                                                                                                                                | 06.25 수 - 06.26 목 (2d) | ```P007-003-Segmentation```        |       | ⬜  |
| ⚙ 기능 구현  | 경량화된 Segmentation 모델을 이용한 옴브레 염색 구현                                                                                                                                                                                                                                                                                                                    | 06.26 목 (1d)           | ```P007-004-Ombre```               |       | ⬜  |
| 📃 문서화   | Segmentation 개발 내용 문서화                                                                                                                                                                                                                                                                                                                                 | 06.25 수 - 06.26 목 (2d) |                                    |       | ⬜  |

**3. LLM 의 AI 윤리 강화**

| 구분         | 계획 내용                                                  | 일정                     | branch                     | issue | 상태 |
|------------|--------------------------------------------------------|------------------------|----------------------------|-------|----|
| 📝 데이터셋 작성 | LLM Supervised Fine-Tuning 학습 데이터 증량 **(AI 윤리 강화)**    | 06.26 목 (1d)           |                            |       | ⬜  |
| 🧪 모델 학습   | LLM Supervised Fine-Tuning 학습 **(Oh-LoRA 답변 생성 LLM만)** | 06.26 목 - 06.27 금 (2d) | ```P007-005-train-LLM```   |       | ⬜  |
| 📝 데이터셋 작성 | BERT (부적절한 언어 여부 파악) 학습 데이터 증량                         | 06.27 금 (1d)           | ```P007-006-BERT-ethics``` |       | ⬜  |
| 🧪 모델 학습   | BERT (부적절한 언어 여부 파악) 학습 (Pre-trained BERT 기반)          | 06.27 금 (1d)           | ```P007-006-BERT-ethics``` |       | ⬜  |
| 📃 문서화     | LLM 개발 내용 문서화                                          | 06.27 금 (1d)           |                            |       | ⬜  |

## 4. 프로젝트 상세 설명

* 사용자의 질문에 대해 **가상 인간 여성 Oh-LoRA 👱‍♀️ (오로라)** 가 답변 생성
  * 이때 주변 환경 및 사용자에 대한 정보 (예: ```[오늘 날씨: 맑음]``` ```[내일 일정: 친구랑 카페 방문]```) 를 Oh-LoRA 의 메모리에 저장
  * Oh-LoRA 는 메모리에 있는 내용 중 가장 관련된 내용을 참고하여 답변
  * 가장 관련된 내용이 없을 경우, **직전 대화 turn 의 요약된 내용을 기억** 하고, **화제 전환 등이 아닌 경우** 그 요약 정보를 참고하여 답변
* Oh-LoRA 의 답변 내용에 따라 가상 인간 여성 이미지 생성
  * **"눈을 뜬 정도, 입을 벌린 정도, 고개 돌림" 의 3가지 속성 값** 을, LLM 의 답변에 기반하여 LLM 으로 생성한 표정 정보 (자연어) 에 따라 적절히 결정

### 4-1. Segmentation 을 이용한 옴브레 염색 구현

* 아래와 같이 **경량화된 Segmentation Model** 을 이용하여 **실시간에 가깝게 Hair 영역 추출**
* 해당 hair 영역 픽셀 정보가 주어지면, 그 정보를 바탕으로 **옴브레 염색** 을 **pre-defined algorithm (NOT AI)** 으로 구현
* 옴브레 염색 적용 대상 Oh-LoRA 얼굴
  * [Oh-LoRA v2 (StyleGAN-VectorFind-v7) 얼굴 27 종](../2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v7/final_OhLoRA_info.md) 
  * [Oh-LoRA v3 (StyleGAN-VectorFind-v8) 얼굴 19 종](../2025_05_26_OhLoRA_v3/stylegan/stylegan_vectorfind_v8/final_OhLoRA_info.md) 

### 4-2. LLM Fine-Tuning 을 이용한 사용자 대화 구현

* 모델을 (TBU) rows 규모의 학습 데이터셋으로 Fine-Tuning
* [RAG (Retrieval Augmented Generation)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md) 과 유사한 컨셉으로 LLM 의 memory 구현
* **Oh-LoRA 👱‍♀️ (오로라) v4 에서 AI 윤리 강화 (부적절한 프롬프트에 대해 응답 거부 & 경고 & 차단 메커니즘 개선)**
* 상세 정보 (TBU)

## 5. 프로젝트 진행 중 이슈 및 해결 방법

* TBU

## 6. 사용자 가이드

* TBU
