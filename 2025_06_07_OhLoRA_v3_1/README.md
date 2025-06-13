
* **LLM을 이용한 답변 생성** 관련 내용은 [Oh-LoRA v3 프로젝트 문서](../2025_05_26_OhLoRA_v3) 및 [v3 프로젝트의 LLM 기술 문서](../2025_05_26_OhLoRA_v3/llm/README.md) 를 참고하시기 바랍니다.

## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. Oh-LoRA 👱‍♀️✨ (오로라) 소개](#1-1-oh-lora--오로라-소개)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 사용한 Python 라이브러리 및 시스템 환경](#2-1-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
  * [4-1. StyleGAN 을 이용한 이미지 생성](#4-1-stylegan-을-이용한-이미지-생성)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)

## 1. 프로젝트 개요

**1. 핵심 아이디어**

* Oh-LoRA 모든 버전 공통 아이디어
  * **LLM Fine-Tuning & StyleGAN** 을 이용한 가상인간 여성 [Oh-LoRA (오로라)](../2025_04_08_OhLoRA) 의 **[2차 업그레이드 버전 (Oh-LoRA v3)](../2025_05_26_OhLoRA_v3) 의 추가 개발**
    * [1차 업그레이드 (Oh-LoRA v2)](../2025_05_02_OhLoRA_v2) 
    * [2차 업그레이드 (Oh-LoRA v3)](../2025_05_26_OhLoRA_v3)
* Oh-LoRA v3.1 아이디어
  * **LLM 추가 개발 없이, 가상 인간 얼굴 생성 부분만 추가 개발** 
  * **w vector (dim=512)** 대신 그 직전의 **$w_{new}$ vector (dim 512 → 2048 로 변경)** 을 이용하여, **핵심 속성 값 (예: 머리 길이, 고개 돌림 정도) 을 정확히 변경시키는 vector 탐색을 보다 효과적으로** 하려고 함

**2. Oh-LoRA 👱‍♀️ (오로라) 이미지 생성 기술**

* [Oh-LoRA v2](../2025_05_02_OhLoRA_v2) 및 [Oh-LoRA v3](../2025_05_26_OhLoRA_v3) 와 같이, 핵심 속성 값 (눈을 뜬 정도, 입을 벌린 정도, 고개 돌림 정도) 을 조정하는 벡터를 찾는 방법 사용
  * [참고 논문 스터디 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)

**3. 참고 사항**

* 본 프로젝트는 제품 출시는 없는 **순수 연구 프로젝트**
* 본 프로젝트의 연구 결과는 **향후 Oh-LoRA 관련 다양한 제품 개발** 에 반영 예정

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

<details><summary>(스포일러) 오로라👱‍♀️ 가 2003년 10월 11일 생인 이유 [ 펼치기 / 접기 ] </summary>

오로라를 개발한 [개발자 (wannabesuperteur)](https://github.com/WannaBeSuperteur) 가 개발할 때 Python 3.10.11 을 사용했기 때문이다.

</details>

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Image Generation (Generative AI)
* 사용 기술

| 기술 분야            | 사용 기술                                                                                                                                                                                         | 설명                                                                                                                                                                                                    |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Image Generation | StyleGAN **(+ Condition Vector Finding)**                                                                                                                                                     | 가상 인간 이미지 생성                                                                                                                                                                                          |
| Image Generation | [SVM (Support Vector Machine)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%AA%A8%EB%8D%B8_SVM.md) | 핵심 속성 값을 변화시키는 벡터를 탐색하기 위한 머신러닝 모델                                                                                                                                                                    |

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

## 3. 프로젝트 일정

* 전체 일정 : **2025.06.07 토 - 06.13 금 (7d)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❎ (DONE BUT **NOT MERGED**), ❌ (FAILED)

**1. 프로젝트 전체 관리**

| 구분     | 계획 내용               | 일정           | branch | issue | 상태 |
|--------|---------------------|--------------|--------|-------|----|
| 📃 문서화 | 프로젝트 개요 및 최초 일정 작성  | 06.07 토 (1d) |        |       | ✅  |
| 📃 문서화 | 모델 HuggingFace 에 등록 | 06.13 금 (1d) |        |       | ✅  |
| 📃 문서화 | 프로젝트 문서 정리 및 마무리    | 06.13 금 (1d) |        |       | 💨 |

**2. StyleGAN 을 이용한 가상 인간 얼굴 생성**

| 구분       | 계획 내용                                                                                                                   | 일정                     | branch                                | issue                                                              | 상태 |
|----------|-------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------|--------------------------------------------------------------------|----|
| 🛠 사전 작업 | [StyleGAN-FineTune-v1](../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan) 구현 | 06.07 토 (1d)           | ```P006-001-StyleGAN-FineTune-v1```   | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/27) | ✅  |
| 🔨 모델 개선 | StyleGAN-FineTune-v9 개발 (w vector dim 512 → 2048)                                                                       | 06.07 토 (1d)           | ```P006-002-StyleGAN-FineTune-v9```   | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/28) | ✅  |
| 🧪 모델 학습 | 필터링된 이미지를 StyleGAN-FineTune-v9 로 추가 Fine-Tuning                                                                         | 06.07 토 - 06.09 월 (3d) | ```P006-002-StyleGAN-FineTune-v9```   | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/28) | ✅  |
| 🔨 모델 개선 | StyleGAN-FineTune-v9 에서 핵심 속성 값만 변화시키는 vector 추출 구현 시, '곱슬머리 or 직모' 핵심 속성 값 추가 **(StyleGAN-VectorFind-v9)**             | 06.10 화 (1d)           | ```P006-003-StyleGAN-VectorFind-v9``` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/29) | ✅  |
| 🧪 모델 학습 | StyleGAN-FineTune-v9 에서 핵심 속성 값만 변화시키는 vector 추출 학습 및 성능 테스트 **(SVM)**                                                  | 06.10 화 - 06.11 수 (2d) | ```P006-003-StyleGAN-VectorFind-v9``` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/29) | ✅  |
| 🧪 모델 학습 | StyleGAN-FineTune-v9 에서 핵심 속성 값만 변화시키는 vector 추출 학습 및 성능 테스트 **(NN + Gradient)**                                        | 06.11 수 - 06.13 금 (3d) | ```P006-003-StyleGAN-VectorFind-v9``` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/29) | ✅  |
| 📃 문서화   | StyleGAN 개발 내용 문서화                                                                                                      | 06.11 수 - 06.13 금 (3d) |                                       |                                                                    | ✅  |

## 4. 프로젝트 상세 설명

* Oh-LoRA 의 답변 내용에 따라 가상 인간 여성 이미지 생성
  * **"눈을 뜬 정도, 입을 벌린 정도, 고개 돌림" 의 3가지 속성 값** 을, LLM 의 답변에 기반하여 LLM 으로 생성한 표정 정보 (자연어) 에 따라 적절히 결정

### 4-1. StyleGAN 을 이용한 이미지 생성

* StyleGAN 의 핵심 속성 값을 변화시키는 벡터를 찾고, 해당 벡터를 이용하는 방법 적용
* [참고 논문 스터디 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)
* [구현 상세 정보](stylegan/README.md)

## 5. 프로젝트 진행 중 이슈 및 해결 방법

* 중요 이슈 없음
