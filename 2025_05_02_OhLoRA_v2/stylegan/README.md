## 목차

* [1. 개요](#1-개요)
  * [1-1. 기존 StyleGAN-FineTune-v4 성능 향상 어려운 원인](#1-1-기존-stylegan-finetune-v4-성능-향상-어려운-원인) 
  * [1-2. StyleGAN-FineTune-v5 개선 방안](#1-2-stylegan-finetune-v5-개선-방안) 
* [2. 핵심 속성 값](#2-핵심-속성-값)
* [3. 사용 모델 설명](#3-사용-모델-설명)
  * [3-1. Fine-Tuned StyleGAN (StyleGAN-FineTune-v5)](#3-1-fine-tuned-stylegan-stylegan-finetune-v5) 
  * [3-2. StyleGAN-FineTune-v1 기반 핵심 속성값 변환 Vector 탐색 (StyleGAN-VectorFind-v6)](#3-2-stylegan-finetune-v1-기반-핵심-속성값-변환-vector-탐색-stylegan-vectorfind-v6) 
* [4. 코드 실행 방법](#4-코드-실행-방법)

## 1. 개요

### 1-1. 기존 StyleGAN-FineTune-v4 성능 향상 어려운 원인

[참고: Oh-LoRA (오로라) 1차 프로젝트 문서](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan)

* 문제 상황
  * 기존 **StyleGAN-FineTune-v4** 의 경우 **StyleGAN-FineTune-v5** 와 마찬가지로 **StyleGAN-FineTune-v1** 을 핵심 속성 값을 이용하여 추가 Fine-Tuning
  * 그러나, **만족할 만한 성능이 나오지 않음**
* 문제 원인 (추정)
  * **Discriminator 구조상의 문제**
    * StyleGAN 의 Discriminator 를 **원래 StyleGAN 의 것으로** 사용
    * 이로 인해, Property CNN 구조처럼 핵심 속성 값을 계산하는 데 특화되어 있지 않음
    * 즉, **Discriminator 의 성능이 충분하지 않아서**, 이와 경쟁하는 Generator 의 성능도 크게 향상되기 어려웠음
  * **Frozen Layer**
    * Discriminator 의 Conv. Layer, Generator 의 Synthesize Layer 등, **Dense Layer 를 제외한 거의 모든 레이어를 Freeze** 처리함
    * 이로 인해 성능이 빠르게 향상되지 않음

### 1-2. StyleGAN-FineTune-v5 개선 방안

## 2. 핵심 속성 값

## 3. 사용 모델 설명

### 3-1. Fine-Tuned StyleGAN (StyleGAN-FineTune-v5)

### 3-2. StyleGAN-FineTune-v1 기반 핵심 속성값 변환 Vector 탐색 (StyleGAN-VectorFind-v6)

* StyleGAN-FineTune-v1 학습 시 latent z vector 512 dim 외에, **원래 label 용도로 추가된 3 dim 을 핵심 속성값 변환 Vector 탐색 목적으로 추가 활용 (총 515 dim)**
  * 해당 3 dim 은 StyleGAN-FineTune-v1 에서는 **16 dim 으로 mapping** 된 후, **latent z dim 512 + 16 → 528 로 concat** 되었음 [(참고)](../../2025_04_08_OhLoRA/stylegan_and_segmentation/model_structure_pdf/stylegan_finetune_v4_generator.pdf)

## 4. 코드 실행 방법

모든 코드는 ```2025_05_02_OhLoRA_v2``` (프로젝트 메인 디렉토리) 에서 실행

* **StyleGAN-FineTune-v5** 모델 Fine-Tuning
  * ```python stylegan/run_stylegan_finetune_v5.py```

* **StyleGAN-VectorFind-v6** 모델을 실행하여 Property Score 를 바꾸는 latent z vector 탐색
  * ```python stylegan/run_stylegan_vectorfind_v6.py```