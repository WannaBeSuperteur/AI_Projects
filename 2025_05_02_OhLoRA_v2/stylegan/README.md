## 목차

* [1. 개요](#1-개요)
  * [1-1. 기존 StyleGAN-FineTune-v4 성능 향상 어려운 원인](#1-1-기존-stylegan-finetune-v4-성능-향상-어려운-원인) 
  * [1-2. StyleGAN-FineTune-v5 개선 방안](#1-2-stylegan-finetune-v5-개선-방안) 
* [2. 핵심 속성 값](#2-핵심-속성-값)
* [3. 사용 모델 설명](#3-사용-모델-설명)
  * [3-1. Fine-Tuned StyleGAN (StyleGAN-FineTune-v5)](#3-1-fine-tuned-stylegan-stylegan-finetune-v5)
  * [3-2. StyleGAN-FineTune-v1 에서 핵심 속성 값 변화시키는 벡터 찾기 (StyleGAN-VectorFind-v6)](#3-2-stylegan-finetune-v1-에서-핵심-속성-값-변화시키는-벡터-찾기-stylegan-vectorfind-v6)
* [4. 코드 실행 방법](#4-코드-실행-방법)

## 1. 개요

* **Oh-LoRA 👱‍♀️ (오로라) 프로젝트의 v2 버전** 에서 사용하는 **가상 인간 여성 이미지 생성 알고리즘**
* **✅ 최종 채택** 알고리즘 : [StyleGAN-VectorFind-v6](#3-2-stylegan-finetune-v1-에서-핵심-속성-값-변화시키는-벡터-찾기-stylegan-vectorfind-v6)
  * StyleGAN-FineTune-v1 (**Fine-Tuned** StyleGAN, **여성 이미지 생성 확률 90% 이상**) 기반
  * StyleGAN-FineTune-v1 의 latent z vector 에서, **[핵심 속성 값](#2-핵심-속성-값) 을 변화시키는 벡터** 를 찾는 아이디어

**전체 모델 파이프라인 그림**

* StyleGAN-VectorFind-v6 을 제외한 나머지 부분은 [오로라 1차 프로젝트의 해당 부분](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md) 과 동일 (해당 문서 참고)

![image](../../images/250502_15.PNG)

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

* Discriminator 구조를 [오로라 1차 프로젝트](../../2025_04_08_OhLoRA/README.md) 의 [Property Score 계산용 CNN](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-3-cnn-model-나머지-핵심-속성-값-7개) 의 구조로 바꾼다.
  * 해당 CNN의 가중치를 먼저 Discriminator 에 주입시킨 후 학습한다.
* Generator 와 Discriminator 의 모든 layer 를 **trainable (학습 가능) 상태** 로 만든다.

## 2. 핵심 속성 값

* 눈을 뜬 정도 ```eyes```, 입을 벌린 정도 ```mouth```, 고개 돌림 정도 ```pose``` 의 3가지 사용
* 상세 정보는 [오로라 1차 프로젝트 문서의 해당 부분](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#2-핵심-속성-값) 참고.
  * 여기서는 핵심 속성 값 계산 알고리즘으로 위 문서에서 언급된 알고리즘 중 [2차 알고리즘](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#2-2-핵심-속성-값-계산-알고리즘-2차-알고리즘-for-stylegan-finetune-v2-v3-v4) 사용

## 3. 사용 모델 설명

| 모델                                                                                                | 최종 채택 | 핵심 아이디어                                                                                                                                                                                                                                                                                                                                                                                                                  | 성능 보고서                                                               |
|---------------------------------------------------------------------------------------------------|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [StyleGAN-FineTune-v5](#3-1-fine-tuned-stylegan-stylegan-finetune-v5)                             | ❌     | - StyleGAN-FineTune-v1 의 **Discriminator 를 [Property Score 계산용 CNN](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-3-cnn-model-나머지-핵심-속성-값-7개) 구조로 바꿔서** 학습<br>- Conditional Truncation 적용 (```trunc_psi``` = 0.5) [(논문 스터디 자료)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.03%5D%20Art%20Creation%20with%20Multi-Conditional%20StyleGANs.md)         | [성능 보고서 **(1차, 3차 모두 성능 미달)**](stylegan_finetune_v5/train_result.md) |
| [StyleGAN-VectorFind-v6](#3-2-stylegan-finetune-v1-에서-핵심-속성-값-변화시키는-벡터-찾기-stylegan-vectorfind-v6) | ✅     | - **핵심 속성값을 잘 변화** 시키는, latent vector z 에 대한 **벡터 찾기** [(논문 스터디 자료)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)<br>- 이때, 이미지를 머리 색 ```hair_color```, 머리 길이 ```hair_length```, 배경색 밝기 평균 ```background_mean``` 에 기반하여 8 그룹으로 나누고, **각 그룹별로 해당 벡터 찾기** | 성능 보고서 **(합격 / TBU)**                                                |                                                    

### 3-1. Fine-Tuned StyleGAN (StyleGAN-FineTune-v5)

**1. 핵심 아이디어**

* StyleGAN-FineTune-v1 의 **Discriminator 를 [Property Score 계산용 CNN](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-3-cnn-model-나머지-핵심-속성-값-7개) 구조로 변경**
  * 처음에 해당 Property Score CNN 의 가중치를 Discriminator 에 주입
* Conditional Truncation 적용 (```trunc_psi``` = 0.5)
  * [해당 스터디 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.03%5D%20Art%20Creation%20with%20Multi-Conditional%20StyleGANs.md) 의 논문의 핵심 아이디어임
* 기타 설정
  * StyleGAN 에서 **Style Mixing 미 적용**
  * 최종 버전 (3차 모델) 의 경우, Generator 와 Discriminator 의 모든 weight 을 학습 가능 (trainable) 처리

![image](../../images/250502_11.PNG)

**2. 성능 보고서**

* [성능 보고서](stylegan_finetune_v5/train_result.md)
* 성능 보고서 요약

| 모델    | ```eyes``` 속성값 | ```mouth``` 속성값    | ```pose``` 속성값 |
|-------|----------------|--------------------|----------------|
| 1차 모델 | 학습이 **전혀 안 됨** | 학습이 **전혀 안 됨**     | 학습이 **거의 안 됨** |
| 3차 모델 | 학습이 **전혀 안 됨** | 학습이 **약간 됨 (불만족)** | 학습이 **거의 안 됨** |

### 3-2. StyleGAN-FineTune-v1 에서 핵심 속성 값 변화시키는 벡터 찾기 (StyleGAN-VectorFind-v6)

## 4. 코드 실행 방법

모든 코드는 ```2025_05_02_OhLoRA_v2``` (프로젝트 메인 디렉토리) 에서 실행

* **StyleGAN-FineTune-v5** 모델 Fine-Tuning
  * ```python stylegan/run_stylegan_finetune_v5.py```