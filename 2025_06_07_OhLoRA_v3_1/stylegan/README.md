## 목차

* [1. 개요](#1-개요)
  * [1-1. 모델 구조](#1-1-모델-구조) 
* [2. 핵심 속성 값](#2-핵심-속성-값)
* [3. 사용 모델 설명](#3-사용-모델-설명)
  * [3-1. Fine-Tuned StyleGAN (StyleGAN-FineTune-v1)](#3-1-fine-tuned-stylegan-stylegan-finetune-v1)
  * [3-2. Fine-Tuned StyleGAN (StyleGAN-FineTune-v9)](#3-2-fine-tuned-stylegan-stylegan-finetune-v9)
  * [3-3. StyleGAN-VectorFind-v9 (SVM 방법)](#3-3-stylegan-vectorfind-v9-svm-방법)
  * [3-4. StyleGAN-VectorFind-v9 (Gradient 방법)](#3-4-stylegan-vectorfind-v9-gradient-방법)
  * [3-5. Gender, Quality, Age, Glass Score CNN (StyleGAN-FineTune-v8 학습 데이터 필터링용)](#3-5-gender-quality-age-glass-score-cnn-stylegan-finetune-v8-학습-데이터-필터링용)
* [4. intermediate vector 추출 위치](#4-intermediate-vector-추출-위치)
* [5. 코드 실행 방법](#5-코드-실행-방법)

## 1. 개요

* 핵심 요약
  * **Oh-LoRA 👱‍♀️ (오로라) 프로젝트의 v3 버전** 에서 사용하는 **가상 인간 여성 이미지 생성 알고리즘**
* 모델 구조 요약
  * Original StyleGAN
  * → StyleGAN-FineTune-v1 **('속성 값' 으로 conditional 한 이미지 생성 시도)**
  * → StyleGAN-FineTune-v9 **(Oh-LoRA 컨셉에 맞는 이미지로 추가 Fine-Tuning)** 
  * → StyleGAN-VectorFind-v9 **(Oh-LoRA 의 표정을 변화시키는 intermediate vector 를 활용)**

### 1-1. 모델 구조

* 전체적으로 [Oh-LoRA v3 (with StyleGAN-VectorFind-v8)](../../2025_05_26_OhLoRA_v3/stylegan/README.md#1-1-모델-구조) 과 유사
* 단, **StyleGAN-FineTune-v8** 대신 **StyleGAN-FineTune-v9** 적용

## 2. 핵심 속성 값

* [Oh-LoRA v3 의 해당 부분](../../2025_05_26_OhLoRA_v3/stylegan/README.md#2-핵심-속성-값) 참고.
* [핵심 속성 값 계산 알고리즘 (Oh-LoRA v3 프로젝트 문서)](../../2025_05_26_OhLoRA_v3/stylegan/README.md#2-1-핵심-속성-값-계산-알고리즘)

## 3. 사용 모델 설명

| 모델                                                                                                                | 최종 채택 | 핵심 아이디어                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 성능 보고서 |
|-------------------------------------------------------------------------------------------------------------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| [StyleGAN-FineTune-v1](#3-1-fine-tuned-stylegan-stylegan-finetune-v1)                                             |       | - StyleGAN-FineTune-v8 모델 학습을 위한 중간 단계 모델                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |        |
| [StyleGAN-FineTune-v9](#3-2-fine-tuned-stylegan-stylegan-finetune-v9)                                             | ✅     | - StyleGAN-FineTune-v1 을 **Oh-LoRA 컨셉에 맞는 이미지** 로 추가 Fine-Tuning 하여, **Oh-LoRA 컨셉에 맞는 이미지 생성 확률 향상**<br>- 즉, 안경을 쓰지 않은, 고품질의 젊은 여성 이미지 생성 확률 향상                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |        |
| [StyleGAN-VectorFind-v9 (SVM 방법)](#3-3-stylegan-vectorfind-v9-svm-방법)                                             |       | - **핵심 속성값을 잘 변화** 시키는, intermediate vector 에 대한 **벡터 찾기** [(논문 스터디 자료)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)<br>- 이때, 이미지를 머리 색 ```hair_color```, 머리 길이 ```hair_length```, 배경색 밝기 평균 ```background_mean```, 직모 vs. 곱슬머리 ```hairstyle```, 에 기반하여 $2^4 = 16$ 그룹으로 나누고, **각 그룹별로 해당 벡터 찾기**<br>- intermediate vector 의 Generator 상의 위치가 [3가지](#4-intermediate-vector-추출-위치) 인 것을 제외하면, [StyleGAN-VectorFind-v8](../../2025_05_26_OhLoRA_v3/stylegan/README.md#3-3-stylegan-finetune-v8-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v8) 과 동일 |
| [StyleGAN-VectorFind-v9 (Gradient 방법)](#3-4-stylegan-vectorfind-v9-gradient-방법)                                   |       | - **핵심 속성값을 잘 변화** 시키는, intermediate vector 에 대한 **벡터 찾기** [(논문 스터디 자료)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)<br>- 이때, intermediate vector 를 입력, ```eyes``` ```mouth``` ```pose``` 핵심 속성 값을 출력으로 하는 **간단한 딥러닝 모델 (Neural Network)** 을 학습, 그 모델을 이용하여 얻은 **Gradient 를 해당 벡터로 간주**                                                                                                                                                                                                                                                                        |
| [Gender, Quality, Age, Glass Score CNN](#3-5-gender-quality-age-glass-score-cnn-stylegan-finetune-v8-학습-데이터-필터링용) |       | - StyleGAN-FineTune-v8 및 v9 모델의 **학습 데이터 필터링** (4개의 [핵심 속성 값](../../2025_05_26_OhLoRA_v3/stylegan/README.md#2-핵심-속성-값) 이용) 을 위한 모델                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |        |

### 3-1. Fine-Tuned StyleGAN (StyleGAN-FineTune-v1)

![image](../../images/250526_1.PNG)

* [오로라 v1 프로젝트](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md) 의 **모든 프로세스 (StyleGAN-FineTune-v1 모델 등) 를 그대로** 사용
* [상세 정보 (오로라 v1 프로젝트 문서)](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan)

### 3-2. Fine-Tuned StyleGAN (StyleGAN-FineTune-v9)

TBU

### 3-3. StyleGAN-VectorFind-v9 (SVM 방법)

TBU

### 3-4. StyleGAN-VectorFind-v9 (Gradient 방법)

TBU

### 3-5. Gender, Quality, Age, Glass Score CNN (StyleGAN-FineTune-v8 학습 데이터 필터링용)

* [Oh-LoRA v3 프로젝트 문서의 해당 부분](../../2025_05_26_OhLoRA_v3/stylegan/README.md#3-4-gender-quality-age-glass-score-cnn-stylegan-finetune-v8-학습-데이터-필터링용) 참고.

## 4. intermediate vector 추출 위치

## 5. 코드 실행 방법

모든 코드는 ```2025_06_07_OhLoRA_v3_1``` (프로젝트 메인 디렉토리) 에서 실행

* **StyleGAN-FineTune-v9** 학습
  * ```python stylegan/run_stylegan_finetune_v9.py```
