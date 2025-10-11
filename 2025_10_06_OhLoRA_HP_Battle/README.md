
## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. Oh-LoRA 👱‍♀️✨ (오로라) 소개](#1-1-oh-lora--오로라-소개)
  * [1-2. 실행 스크린샷](#1-2-실행-스크린샷)
  * [1-3. Oh-LoRA 얼굴 변화 애니메이션](#1-3-oh-lora-얼굴-변화-애니메이션)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 사용한 Python 라이브러리 및 시스템 환경](#2-1-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
  * [4-1. 선정 데이터셋](#4-1-선정-데이터셋)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. 사용자 가이드](#6-사용자-가이드)

## 1. 프로젝트 개요

**1. 핵심 아이디어**

* **LLM Fine-Tuning & StyleGAN** 을 이용한 가상인간 여성 [Oh-LoRA (오로라)](../2025_04_08_OhLoRA) 을 이용한 **인간과의 하이퍼파라미터 대결 컨셉**
* **특정 이미지 분류 데이터셋** 대상으로, Oh-LoRA 👱‍♀️ 가 최적의 하이퍼파라미터 탐색 및 **인간이 탐색한 하이퍼파라미터 결과와의 성능 비교**

----

<details><summary>Oh-LoRA 👱‍♀️ (오로라) 얼굴 생성 방법 [ 펼치기 / 접기 ]</summary>

* 다음과 같이 **StyleGAN-VectorFind-v7** 및 **StyleGAN-VectorFind-v8** 을 이용하여 **Oh-LoRA 👱‍♀️ (오로라)** 얼굴 이미지 생성
  * [StyleGAN-VectorFind-v7](../2025_05_02_OhLoRA_v2/stylegan/README.md#3-3-stylegan-finetune-v1-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v7)
  * [StyleGAN-VectorFind-v8](../2025_05_26_OhLoRA_v3/stylegan/README.md#3-3-stylegan-finetune-v8-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v8)
* **StyleGAN-VectorFind-v7** 구조 개념도

![image](../images/250502_23.PNG)

* **StyleGAN-VectorFind-v8** 구조 개념도

![image](../images/250526_12.png)

* 실제 생성되는 얼굴
  * [Oh-LoRA v2 (StyleGAN-VectorFind-v7) 얼굴 27 종](../2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v7/final_OhLoRA_info.md) 
  * [Oh-LoRA v3 (StyleGAN-VectorFind-v8) 얼굴 19 종](../2025_05_26_OhLoRA_v3/stylegan/stylegan_vectorfind_v8/final_OhLoRA_info.md) 

</details>

----

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
  * 사람이 아닌 AI 가상 인간, 즉 **AI 요정** 만이 가질 수 있음

<details><summary>(스포일러) 오로라👱‍♀️ 가 2003년 10월 11일 생인 이유 [ 펼치기 / 접기 ] </summary>

오로라를 개발한 [개발자 (wannabesuperteur)](https://github.com/WannaBeSuperteur) 가 개발할 때 Python 3.10.11 을 사용했기 때문이다.

</details>

### 1-2. 실행 스크린샷

TBU

### 1-3. Oh-LoRA 얼굴 변화 애니메이션

* [해당 문서](../2025_06_24_OhLoRA_v4/ohlora_animation.md) 참고.
* **총 20 MB 정도의 GIF 이미지 (10장) 가 있으므로 데이터 사용 시 주의**

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Computer Vision 
  * LLM (Large Language Model)
  * Meta Model (AutoML?)
* 사용 기술

| 기술 분야           | 사용 기술                                                                                                                                                        | 설명                                         |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| LLM             | [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)    | 가상 인간이 적절한 말투로 사용자와의 대결 결과를 표현할 수 있게 하는 기술 |
| LLM             | [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md) | 가상 인간의 LLM 을 효율적으로 Fine-Tuning 하는 기술       |
| Computer Vision | CNN (Conv. NN)                                                                                                                                               | 기본적인 이미지 처리                                |
| Meta Model      | Hyperparameter Optimization                                                                                                                                  | Oh-LoRA 👱‍♀️ 의 모델을 이용한 자동화된 최적 하이퍼파라미터 탐색 |

### 2-1. 사용한 Python 라이브러리 및 시스템 환경

* Python
  * Python : **Python 3.10.11**
  * Dev Tool : PyCharm 2024.1 Community Edition
* Python Libraries
  * [주요 파이썬 라이브러리](system_info_and_user_guide.md#1-1-주요-python-라이브러리)
  * [실험 환경의 전체 파이썬 라이브러리 목록](system_info_and_user_guide.md#1-2-시스템에-설치된-전체-python-라이브러리)
* OS & CPU & GPU
  * OS : **Windows 10**
  * CPU : Intel(R) Xeon(R) CPU E5-2690 0 @ 2.90GHz
  * GPU : 2 x **Quadro M6000** (12 GB each)
  * **CUDA 12.4** (NVIDIA-SMI 551.61)
* [시스템 환경 상세 정보](system_info_and_user_guide.md#1-시스템-환경)

## 3. 프로젝트 일정

* 전체 일정 : **2025.10.06 월 - 10.15 수**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❎ (DONE BUT **NOT MERGED**), ❌ (FAILED)

**1. 프로젝트 전체 관리**

| 구분       | 계획 내용                      | 일정           | branch                 | issue | 상태 |
|----------|----------------------------|--------------|------------------------|-------|----|
| 📃 문서화   | 프로젝트 개요 및 최초 일정 작성         | 10.06 월 (1d) |                        |       | ✅  |
| 🔍 최종 검토 | 최종 사용자 실행용 코드 작성           | 10.14 화 (1d) | ```P009-007-ForUser``` |       | ⬜  |
| 🔍 최종 검토 | 최종 QA (버그 유무 검사)           | 10.15 수 (1d) |                        |       | ⬜  |
| 📃 문서화   | 데이터셋 및 모델 HuggingFace 에 등록 | 10.15 수 (1d) |                        |       | ⬜  |
| 📃 문서화   | 프로젝트 문서 정리 및 마무리           | 10.15 수 (1d) |                        |       | ⬜  |

**2. 하이퍼파라미터 튜닝 대결**

| 구분         | 계획 내용                                                 | 일정                     | branch                               | issue | 상태 |
|------------|-------------------------------------------------------|------------------------|--------------------------------------|-------|----|
| 📝 데이터셋 선택 | 인간과의 하이퍼파라미터 대결을 할 **이미지 분류 데이터셋** 선정                 | 10.11 토 (1d)           |                                      |       | ✅  |
| 🔨 모델 구현   | 기본 CNN 모델 (하이퍼파라미터 튜닝 대상) 구현                          | 10.11 토 (1d)           | ```P009-001-CNN```                   |       | ⬜  |
| 🔨 모델 구현   | 이미지의 hidden representation 모델 구현 (Auto-Encoder?)      | 10.11 토 (1d)           | ```P009-002-hidden-representation``` |       | ⬜  |
| 🧪 모델 학습   | 이미지의 hidden representation 모델 학습                      | 10.11 토 (1d)           | ```P009-002-hidden-representation``` |       | ⬜  |
| 🔨 모델 구현   | 최적 하이퍼파라미터 탐색 모델 구현 **(hidden representation 입력 기반)** | 10.11 토 (1d)           | ```P009-003-hp-find-model```         |       | ⬜  |
| 🧪 모델 학습   | 최적 하이퍼파라미터 탐색 모델 학습                                   | 10.11 토 - 10.13 월 (3d) | ```P009-004-hp-find-model-train```   |       | ⬜  |
| 📃 문서화     | "하이퍼파라미터 튜닝 대결" 개발 내용 문서화                             | 10.13 월 (1d)           |                                      |       | ⬜  |

**3. LLM을 이용한 대결 결과 표현**

| 구분       | 계획 내용                                                | 일정           | branch                       | issue | 상태 |
|----------|------------------------------------------------------|--------------|------------------------------|-------|----|
| 🧪 모델 학습 | 대결 결과 표현 LLM Supervised Fine-Tuning 학습<br>(LLM: ???) | 10.13 월 (1d) | ```P009-005-train-LLM```     |       | ⬜  |
| ⚙ 기능 구현  | 대결 결과 리스트 기반 결과 표현 출력 구현                             | 10.14 화 (1d) | ```P009-006-battle-result``` |       | ⬜  |
| 📃 문서화   | "LLM을 이용한 대결 결과 표현" 개발 내용 문서화                        | 10.14 화 (1d) |                              |       | ⬜  |

## 4. 프로젝트 상세 설명

### 4-1. 선정 데이터셋

**1. 데이터셋 선정 기준**

* **일정 규모 이상의 이미지 분류 데이터셋**
* 머신러닝 분야에서 널리 알려진 데이터셋
* 본 프로젝트에 적용하기 적합한 라이선스의 데이터셋

**2. 선정 데이터셋**

| 데이터셋                                                                                      | 설명                                            | 이미지 해상도<br>(Channel 개수 포함) | Class 개수 | train data | test data |
|-------------------------------------------------------------------------------------------|-----------------------------------------------|----------------------------|----------|------------|-----------|
| MNIST [(download)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)             | 딥러닝 초보자의 연습용 & 모델 설명 예시용으로 널리 사용되는 숫자 분류 데이터셋 | 1 x 28 x 28                | 10       | 60,000     | 10,000    |
| Fashion MNIST [(download)](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | MNIST와 유사하되, 숫자가 아닌 옷 이미지를 사용                 | 1 x 28 x 28                | 10       | 60,000     | 10,000    |
| CIFAR-10 [(download)](https://www.kaggle.com/datasets/ayush1220/cifar10)                  | Object 분류 데이터셋                                | 3 x 32 x 32                | 10       | 50,000     | 10,000    |

* citation for CIFAR-10 dataset
  * [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

## 5. 프로젝트 진행 중 이슈 및 해결 방법

TBU

## 6. 사용자 가이드

TBU