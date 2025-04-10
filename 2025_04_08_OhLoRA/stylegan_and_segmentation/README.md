## 목차

* [1. 개요](#1-개요)
* [2. 핵심 속성 값](#2-핵심-속성-값)
  * [2-1. 핵심 속성 값 계산 알고리즘](#2-1-핵심-속성-값-계산-알고리즘)
* [3. 사용 모델 설명](#3-사용-모델-설명)
  * [3-1. Image Generation Model (StyleGAN)](#3-1-image-generation-model-stylegan)
  * [3-2. Segmentation Model](#3-2-segmentation-model)
* [4. 코드 실행 방법](#4-코드-실행-방법)

## 1. 개요

* 가상 인간 이미지 생성을 위한 StyleGAN 구현
* 기본 StyleGAN 으로 이미지 생성 후, Pre-trained Segmentation 모델을 이용하여 **핵심 속성 값** 을 라벨링 
  * 단, 성별은 생성된 이미지 중 2,000 장에 대해 **수기로 라벨링** 하여 CNN 으로 학습 후, 해당 CNN 으로 나머지 8,000 장에 대해 성별 값 추정
  * 생성 대상 이미지가 여성이므로, **여성 이미지만을 따로 필터링한 후, 필터링된 이미지에 대해서만 나머지 속성을 라벨링** 
* 핵심 속성 값에 해당하는 element 를 StyleGAN 의 latent vector 에 추가하여, 생성된 이미지에 대해 Fine-Tuning 실시
  * 이때, latent vector 와 관련된 부분을 제외한 나머지 StyleGAN parameter 는 모두 freeze

![image](../../images/250408_1.PNG)

## 2. 핵심 속성 값

| 핵심 속성 값 이름                     | 설명                                | 값 범위    |
|--------------------------------|-----------------------------------|---------|
| 성별 ```gender```                | 0 (남성) ~ 1 (여성) 의 확률 값            | 0 ~ 1   |
| 눈을 뜬 정도 ```eyes```             | 눈을 크게 뜰수록 값이 큼                    | 0 ~ 1   |
| 머리 색 ```hair_color```          | 머리 색이 밝을수록 값이 큼                   | 0 ~ 1   |
| 입을 벌린 정도 ```mouth```           | 입을 벌린 정도가 클수록 값이 큼                | 0 ~ 1   |
| 배경색의 밝기 ```background_light``` | 배경색이 밝을수록 값이 큼                    | 0 ~ 1   |
| 배경색의 표준편차 ```background_std``` | 배경색의 표준편차가 클수록 값이 큼               | 0 ~ 1   |
| 얼굴의 위치 ```face_pose```         | 왼쪽 쳐다봄 (-1), 정면 (0), 오른쪽 쳐다봄 (+1) | -1 ~ +1 |

### 2-1. 핵심 속성 값 계산 알고리즘

* Segmentation 결과를 바탕으로 다음과 같이 **핵심 속성 값들을 계산**
  * TBU 

## 3. 사용 모델 설명

### 3-1. Image Generation Model (StyleGAN)

[Implementation & Pre-trained Model Source : GenForce GitHub](https://github.com/genforce/genforce/tree/master) (MIT License)

* Generator
  * ```stylegan/stylegan_generator.py```
* Discriminator
  * ```stylegan/stylegan_discriminator.py```
* Model
  * ```stylegan/stylegan_model.pth``` (Original GAN, including Generator & Discriminator)
    * from [MODEL ZOO](https://github.com/genforce/genforce/blob/master/MODEL_ZOO.md) > StyleGAN Ours > **celeba-partial-256x256**

### 3-2. Segmentation Model

## 4. 코드 실행 방법

**모든 코드는 ```2025_04_08_OhLoRA``` main directory 에서 실행**

* Original GAN Generator 실행하여 이미지 생성
  * ```python stylegan_and_segmentation/run_original_generator.py```
