## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. 성능 향상 방법](#1-1-성능-향상-방법)
  * [1-2. 성능 향상 결과](#1-2-성능-향상-결과)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 관련 논문](#2-1-관련-논문)
  * [2-2. 사용한 Python 라이브러리 및 시스템 환경](#2-2-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. inference 실행 가이드](#6-inference-실행-가이드)

## 1. 프로젝트 개요

* **Medical Segmentation** 데이터셋 중 하나인 **Kvasir-SEG (gastrointestinal polyp, 위장관 용종)** 데이터셋에서 우수한 성능을 거둔 모델 중 하나로 **EffiSegNet** 이 있다. ([PapersWithCode 참고](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg))
  * [Kvasir-SEG 데이터셋](https://datasets.simula.no/kvasir-seg/) (상업적 사용을 위해서는 별도 허가 필요)
  * [EffiSegNet Official Code (PyTorch)](https://github.com/ivezakis/effisegnet)

| 모델            | mean Dice       | mIoU                |
|---------------|-----------------|---------------------|
| EffiSegNet-B5 | 0.9488 (rank 2) | **0.9065 (rank 1)** |
| EffiSegNet-B4 | 0.9483 (rank 3) | 0.9056 (rank 2)     |

* 본 프로젝트는 **EffiSegNet 으로 Kvasir-SEG 데이터셋을 학습** 시켰을 때,
  * **'오답'이 발생하는 부분** 을 찾고,
  * 그 오답의 원인과 해결 방법을 탐색 및 적용하여 성능을 향상시키는 것을 목표로 한다.

### 1-1. 성능 향상 방법

* Augmentation 조정
  * ColorJitter & Affine 변환에 대한 실시 확률 각각 **50% → 80%** 로 증가 [(필요성)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Image%20Processing/Basics_Image_Augmentation_Methods.md#2-torchvision-%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-augmentation) [(details)](effisegnet_improved/README.md#2-1--colorjitter--affine-prob-상향) 
  * ColorJitter Augmentation 강도 약화 [(details)](effisegnet_improved/README.md#2-3--weaken-colorjitter)
* 이미지 좌측 상단에 검은색 직사각형 추가
  * Kvasir-SEG 데이터셋에 특화된 Augmentation [(details)](effisegnet_improved/README.md#2-5--black-rectangle-추가)
* Near-Pixel-Diff Loss Term 추가
  * Segmentation Map 의 noise 에 의해 정답과 오차가 생기는 현상 해결 [(details)](effisegnet_improved/README.md#2-6--near-pixel-diff-loss-term-추가)

### 1-2. 성능 향상 결과

![image](../images/250522_2.png)

| 구분                                              | Test Dice Score       | Test IoU Score        | Test Recall           | 
|-------------------------------------------------|-----------------------|-----------------------|-----------------------|
| **Original Model**                              | 0.9310                | 0.8803                | 0.9363                |
| Augmentation 조정 적용 모델<br>- 4차 수정 (05.24)        | **0.9421 (▲ 0.0111)** | **0.8944 (▲ 0.0141)** | 0.9385 (▲ 0.0022)     |
| 이미지 좌측 상단 검은색 직사각형 추가<br>- 5차 수정 (05.25)        | 0.9370 (▲ 0.0060)     | 0.8879 (▲ 0.0076)     | 0.9378 (▲ 0.015)      |
| Near-Pixel-Diff Loss Term 추가<br>- 7차 수정 (05.25) | 0.9347 (▲ 0.0037)     | 0.8824 (▲ 0.0021)     | **0.9528 (▲ 0.0165)** |

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Computer Vision
  * LLM (optional)
* 사용 기술

| 기술 분야           | 사용 기술      | 설명                         |
|-----------------|------------|----------------------------|
| Computer Vision | EffiSegNet | Kvasir-SEG 데이터셋에서의 병변 등 탐지 |

### 2-1. 관련 논문

본 프로젝트에서 사용할 기술에 대한 **탄탄한 기초가 중요하다** 는 판단 아래 작성한, 관련 논문에 관한 스터디 자료이다.

* **EffiSegNet** (Segmentation Model) 논문
  * [논문 : EffiSegNet: Gastrointestinal Polyp Segmentation through a Pre-Trained EfficientNet-based Network with a Simplified Decoder (2024.07)](https://arxiv.org/pdf/2407.16298v1)
  * [논문 스터디 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.22%5D%20EffiSegNet%20-%20Gastrointestinal%20Polyp%20Segmentation%20through%20a%20Pre-Trained%20EfficientNet-based%20Network%20with%20a%20Simplified%20Decoder.md)

### 2-2. 사용한 Python 라이브러리 및 시스템 환경

* Python
  * Python : **Python 3.10.11**
  * Dev Tool : PyCharm 2024.1 Community Edition
* Python Libraries
  * 주요 파이썬 라이브러리 (TBU)
  * 실험 환경의 전체 파이썬 라이브러리 목록 (TBU)
* OS & GPU
  * OS : **Windows 10**
  * GPU : 2 x **Quadro M6000** (12 GB each)
  * **CUDA 12.4** (NVIDIA-SMI 551.61)

## 3. 프로젝트 일정

* 전체 일정 : **2025.05.22 목 - 05.26 월 (5d)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❎ (DONE BUT **NOT MERGED**), ❌ (FAILED)

| 구분       | 계획 내용                                 | 일정                     | branch                    | issue                                                              | 상태 |
|----------|---------------------------------------|------------------------|---------------------------|--------------------------------------------------------------------|----|
| 📃 문서화   | 프로젝트 개요 및 최초 일정 작성                    | 05.22 목 (1d)           |                           |                                                                    | ✅  |
| 📕 논문    | 논문 탐독 (EffiSegNet)                    | 05.22 목 (1d)           |                           |                                                                    | ✅  |
| 🔨 모델 구현 | EffiSegNet 구현                         | 05.23 금 (1d)           | ```P004-001-EffiSegNet``` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/11) | ✅  |
| 🧪 모델 학습 | EffiSegNet 학습 **1차** (원본 구현)          | 05.23 금 - 05.24 토 (2d) | ```P004-001-EffiSegNet``` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/11) | ✅  |
| 🔬 탐구    | EffiSegNet 의 '오답'이 발생하는 부분 탐구         | 05.24 토 (1d)           |                           | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/12) | ✅  |
| 🔨 모델 개선 | EffiSegNet 개선                         | 05.24 토 - 05.25 일 (2d) | ```P004-002-EffiSegNet``` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/13) | ✅  |
| 🧪 모델 학습 | EffiSegNet 학습 **2차** (개선된 모델)         | 05.24 토 - 05.25 일 (2d) | ```P004-002-EffiSegNet``` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/13) | ✅  |
| 📃 문서화   | EffiSegNet 원본 vs 개선 모델 성능 비교 및 보고서 작성 | 05.26 월 (1d)           |                           |                                                                    | 💨 |
| 📃 문서화   | 프로젝트 문서 정리 및 마무리                      | 05.26 월 (1d)           |                           |                                                                    | ⬜  |

## 4. 프로젝트 상세 설명

* EffiSegNet baseline 모델 구현
  * [상세 정보](effisegnet_base/README.md)

| 구분                                    | Dice Score | IoU Score  | 
|---------------------------------------|------------|------------|
| Original Paper                        | 0.9483     | 0.9056     |
| **test (50 epochs)**                  | **0.9445** | **0.8980** |
| test (300 epochs = Original Paper 조건) | 0.9413     | 0.8965     |

* EffiSegNet improved 모델 구현
  * [상세 정보](effisegnet_improved/README.md) 
  * 암 (Positive) 인데 암이 아니라고 예측 (Negative) 하는 오류는 **비교적 심각한 문제** 이므로, 이를 고려한 성능지표인 Recall 이 중요
  * **코드는 분명 동일한데, 50 epoch test 시 결과가 현저히 차이 나는 이유는 불명**

| 구분                                                  | Dice Score            | IoU Score             | Recall                | 
|-----------------------------------------------------|-----------------------|-----------------------|-----------------------|
| **Original Model test (50 epochs)**                 | 0.9310                | 0.8803                | 0.9363                |
| Best Improved Model (Dice & IoU)<br>- 4차 수정 (05.24) | **0.9421 (▲ 0.0111)** | **0.8944 (▲ 0.0141)** | 0.9385 (▲ 0.0022)     |
| Best Improved Model (Recall)<br>- 7차 수정 (05.25)     | 0.9347 (▲ 0.0037)     | 0.8824 (▲ 0.0021)     | **0.9528 (▲ 0.0165)** |

## 5. 프로젝트 진행 중 이슈 및 해결 방법

## 6. inference 실행 가이드

* **1. Kvasir-SEG 데이터셋 다운로드**
  * from [EffiSegNet Official GitHub](https://github.com/ivezakis/effisegnet/tree/main/Kvasir-SEG)

* **2. 실험 차수 checkout**
  * 먼저, 아래에서 원하는 실험 차수의 revision 으로 checkout

| 실험 차수               | Test Dice  | Test IoU   | Test Recall | checkout 할 revision                                                                                         |
|---------------------|------------|------------|-------------|-------------------------------------------------------------------------------------------------------------|
| Original EffiSegNet | 0.9310     | 0.8803     | 0.9363      | 최신 버전                                                                                                       |
| 1차 수정 (05.24)       | 0.9406     | 0.8913     | 0.9259      | [eb0766bd](https://github.com/WannaBeSuperteur/AI_Projects/commit/eb0766bd36015ab3de3ac58f5d47f4c1771bcfdf) |
| 2차 수정 (05.24)       | 0.9363     | 0.8860     | 0.9295      | [14339e3a](https://github.com/WannaBeSuperteur/AI_Projects/commit/14339e3a03419c04178a6ddcda18008ae2c7c516) |
| 3차 수정 (05.24)       | 0.9386     | 0.8904     | 0.9389      | [c3fd8c03](https://github.com/WannaBeSuperteur/AI_Projects/commit/c3fd8c03a50c72f92b89433802fab96cf045ec91) |
| 4차 수정 (05.24)       | **0.9421** | **0.8944** | 0.9385      | [9954c2b6](https://github.com/WannaBeSuperteur/AI_Projects/commit/9954c2b659b492d2250fec394acfed48d3c73b76) |
| 5차 수정 (05.24)       | 0.9370     | 0.8879     | 0.9378      | [9ef7b411](https://github.com/WannaBeSuperteur/AI_Projects/commit/9ef7b411b4f653e65977349b5c2410cdd499bcf1) |
| 6차 수정 (05.25)       | 0.9304     | 0.8790     | 0.9295      | [1378c41a](https://github.com/WannaBeSuperteur/AI_Projects/commit/1378c41a972931a6e3b31e04cf5964e47ac3f773) |
| 7차 수정 (05.25)       | 0.9347     | 0.8824     | **0.9528**  | [63a1183f](https://github.com/WannaBeSuperteur/AI_Projects/commit/63a1183fcad8820ccd002fb2ffa5ff95829a19dc) |
| 8차 수정 (05.25)       | 0.9315     | 0.8772     | 0.9501      | [5737351c](https://github.com/WannaBeSuperteur/AI_Projects/commit/5737351c5a70bf024380997f7af4b75a89fab8c5) |

* **3. 학습 및 inference 코드 실행**
  * ```python effisegnet_improved/train.py```
  * Quadro M6000 12GB 기준 **train & final test (inference)** 까지 **약 1h 40m 소요**