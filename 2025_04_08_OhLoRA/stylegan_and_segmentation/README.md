## 목차

* [1. 개요](#1-개요)
  * [1-1. 품질 판단이 필요한 이유](#1-1-품질-판단이-필요한-이유) 
* [2. 핵심 속성 값](#2-핵심-속성-값)
  * [2-1. 핵심 속성 값 계산 알고리즘](#2-1-핵심-속성-값-계산-알고리즘)
* [3. 사용 모델 설명](#3-사용-모델-설명)
  * [3-1. Image Generation Model (StyleGAN)](#3-1-image-generation-model-stylegan)
  * [3-2. CNN Model](#3-2-cnn-model)
  * [3-3. Segmentation Model (FaceXFormer)](#3-3-segmentation-model-facexformer)
* [4. 코드 실행 방법](#4-코드-실행-방법)

## 1. 개요

* 가상 인간 이미지 생성을 위한 StyleGAN 구현
* 기본 StyleGAN 으로 이미지 생성 후, Pre-trained Segmentation 모델을 이용하여 **핵심 속성 값** 을 라벨링 
  * 단, 성별은 생성된 이미지 중 2,000 장에 대해 **수기로 라벨링** 하여 CNN 으로 학습 후, 해당 CNN 으로 나머지 8,000 장에 대해 성별 값 추정
  * 생성 대상 이미지가 여성이므로, **여성 이미지 (품질이 나쁜 이미지 제외) 만을 따로 필터링한 후, 필터링된 이미지에 대해서만 나머지 속성을 라벨링** 
* 핵심 속성 값에 해당하는 element 를 StyleGAN 의 latent vector 에 추가하여, 생성된 이미지에 대해 Fine-Tuning 실시
  * 이때, latent vector 와 관련된 부분을 제외한 나머지 StyleGAN parameter 는 모두 freeze

![image](../../images/250408_1.PNG)

### 1-1. 품질 판단이 필요한 이유

* StyleGAN 으로 생성한 이미지의 약 5% 가 아래와 같이 **저품질의 이미지** 임
  * 저품질 이미지의 경우, 보통 **이미지 상단 부분에 blur** 가 발생하여 어색함 
* 저품질 이미지 생성을 원천 방지하기 위하여, **StyleGAN Fine-Tuning 학습 데이터에서 저품질의 이미지를 제외** 할 필요가 있음

![image](../../images/250408_3.PNG)

## 2. 핵심 속성 값

* 성별, 이미지 품질을 제외한 나머지 핵심 속성 값들은 아래 표와 같이, **계산된 값을 최종적으로 $N(0, 1^2)$ 로 정규화** 하여 AI 모델에 적용
* 필터링에 사용
  * **해당 값이 모두 threshold 에 도달하는 이미지만** 따로 필터링하여, 나머지 7가지 속성 값을 Pre-trained Segmentation Model 을 이용하여 계산

| 핵심 속성 값 이름                    | 설명                                    | AI 모델에 저장되는 값의 범위 또는 분포<br>(CNN output or StyleGAN latent vector) | 필터링에 사용 |
|-------------------------------|---------------------------------------|-------------------------------------------------------------------|---------|
| 성별 ```gender```               | 0 (남성) ~ 1 (여성) 의 확률 값                | 0 ~ 1                                                             | **O**   |
| 이미지 품질 ```quality```          | 0 (저품질) ~ 1 (고품질) 의 확률 값              | 0 ~ 1                                                             | **O**   |
| 눈을 뜬 정도 ```eyes```            | 눈을 크게 뜰수록 값이 큼                        | $N(0, 1^2)$                                                       | X       |
| 머리 색 ```hair_color```         | 머리 색이 밝을수록 값이 큼                       | $N(0, 1^2)$                                                       | X       |
| 머리 길이 ```hair_length```       | 머리 길이가 길수록 값이 큼                       | $N(0, 1^2)$                                                       | X       |
| 입을 벌린 정도 ```mouth```          | 입을 벌린 정도가 클수록 값이 큼                    | $N(0, 1^2)$                                                       | X       |
| 고개 돌림 ```face_pose```         | 왼쪽 고개 돌림 (-1), 정면 (0), 오른쪽 고개 돌림 (+1) | $N(0, 1^2)$                                                       | X       |
| 배경색 평균 ```background_mean```  | 이미지 배경 부분 픽셀의 색의 평균값이 클수록 값이 큼        | $N(0, 1^2)$                                                       | X       |
| 배경색 표준편차 ```background_std``` | 이미지 배경 부분 픽셀의 색의 표준편차가 클수록 값이 큼       | $N(0, 1^2)$                                                       | X       |

### 2-1. 핵심 속성 값 계산 알고리즘

* Segmentation 결과를 바탕으로 다음과 같이 **성별, 이미지 품질을 제외한 7가지 핵심 속성 값들을 계산**
  * 계산 대상 핵심 속성 값 
    * 눈을 뜬 정도, 머리 색, 머리 길이, 입을 벌린 정도, 얼굴의 위치, 배경색 평균, 배경색 표준편차
  * 점수 계산 완료 후, **모든 이미지에 대해 각 속성 종류별로 그 값들을 위 표에 따라 [Gaussian Normalization](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4_%EA%B8%B0%EC%B4%88_Normalization.md#2-2-standarization-z-score-normalization) 적용**
    * 예를 들어, 모든 이미지에 대한 머리 색의 값이 ```[100, 250, 120, 180, 210]``` 인 경우, 이를 Gaussian Normalization 하여 ```[-1.294, 1.402, -0.935, 0.144, 0.683]``` 으로 정규화
  * Segmentation 결과는 **224 x 224 로 resize 된 이미지** 임

**1. 눈을 뜬 정도 (eyes)**

* Segmentation 결과에서 왼쪽 눈과 오른쪽 눈에 해당하는 픽셀들을 각각 찾아서,
* 그 y좌표의 최댓값과 최솟값의 차이를 눈 영역의 높이, 즉 눈을 뜬 정도로 간주

![image](../../images/250408_5.PNG)

**2. 머리 색 (hair_color), 머리 길이 (hair_length)**

* 머리 길이의 경우,
  * Segmentation 결과에서 **hair 영역이 맨 아래쪽 row (224 번째 row) 까지** 있으면,
  * 그 아래쪽의 머리 길이를 **맨 아래쪽 row (1 x 224) 의 hair 영역 픽셀 개수** 를 근거로 예측하는 알고리즘 적용

![image](../../images/250408_6.PNG)

**3. 입을 벌린 정도 (mouth), 고개 돌림 (pose)**

* 고개를 돌리면 코에 해당하는 픽셀의 x 좌표 및 y 좌표 분포에도 영향을 미치므로, 그 상관계수를 이용
  * 코에 해당하는 x 좌표와 y 좌표의 **상관계수가 +** 이면 고개 돌림 방향은 **왼쪽 (Pose Score < 0)**
  * 코에 해당하는 x 좌표와 y 좌표의 **상관계수가 -** 이면 고개 돌림 방향은 **오른쪽 (Pose Score > 0)**

![image](../../images/250408_7.PNG)

**4. 배경색 평균 (background_mean), 배경색 표준편차 (background_std)**

* 이미지 위쪽 절반의 픽셀 중 배경에 해당하는 픽셀 값의 R,G,B 평균값에 대해,
* 그 값들 중 **상위 5% 및 하위 5%를 제외한 90% (잘못된 Segmentation 에 Robust 하도록)** 의 평균 및 표준편차를 각각 의미

![image](../../images/250408_8.PNG)

## 3. 사용 모델 설명

| 모델                      | 모델 분류                          | 사용 목적                                  |
|-------------------------|--------------------------------|----------------------------------------|
| **Original** StyleGAN   | Image Generation Model         | StyleGAN 의 Fine-Tuning 에 사용할 후보 이미지 생성 |
| CNN                     |                                | StyleGAN Fine-Tuning 후보 이미지의 필터링       |
| FaceXFormer             | Pre-trained Segmentation Model | 필터링된 후보 이미지의 핵심 속성 값 추출                |
| **Fine-Tuned** StyleGAN | Image Generation Model         | **Oh-LoRA (오로라) 이미지 생성용 최종 모델**        |

### 3-1. Image Generation Model (StyleGAN)

[Implementation & Pre-trained Model Source : GenForce GitHub](https://github.com/genforce/genforce/tree/master) (MIT License)

| 모델                   | 설명                                                                                               | 여성 이미지 생성                        | 핵심 속성값 오류 없음 | 핵심 속성값 의미 반영 생성 |
|----------------------|--------------------------------------------------------------------------------------------------|----------------------------------|--------------|-----------------|
| Original StyleGAN    | [GenForce GitHub](https://github.com/genforce/genforce/tree/master) 에서 다운받은 Pre-trained StyleGAN | ❌ (**여성 55.6%** = 1,112 / 2,000) | ❌            | ❌               |
| StyleGAN-FineTune-v1 | Original StyleGAN 으로 생성한 여성 이미지 4,703 장으로 Fine-Tuning 한 StyleGAN                                 | ✅ (**여성 93.7%** = 281 / 300)     | ❌            | ❌               |
| StyleGAN-FineTune-v2 | StyleGAN-FineTune-v1 을 **CNN을 포함한 신경망** 으로 추가 학습                                                 |                                  |              |                 |
| StyleGAN-FineTune-v3 | StyleGAN-FineTune-v1 을 **Conditional VAE** 의 Decoder 로 사용하여 추가 학습                                |                                  |              |                 |

* 핵심 속성값 오류
  * 핵심 속성 값 (성별, 이미지 품질 제외 7가지) 이 달라지면 **동일한 인물의 특징이 달라지는 것이 아닌, 아예 다른 인물이 생성되는** 것
* 핵심 속성값 의미 반영 생성
  * 핵심 속성값의 의미 (눈을 뜬 정도, 입을 벌린 정도, 머리 색, 머리 길이, 배경 정보 등) 를 반영하여 인물 이미지가 생성되는지의 여부

**1. Original Model**

* Generator
  * ```stylegan/stylegan_generator.py```
* Discriminator
  * ```stylegan/stylegan_discriminator.py```
* Model Save Path
  * ```stylegan/stylegan_model.pth``` (**Original GAN**, including Generator & Discriminator)
    * original model from [MODEL ZOO](https://github.com/genforce/genforce/blob/master/MODEL_ZOO.md) > StyleGAN Ours > **celeba-partial-256x256**
* [Study Doc (2025.04.09)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.04.09%5D%20A%20Style-Based%20Generator%20Architecture%20for%20Generative%20Adversarial%20Networks.md)

**2. Modified Fine-Tuned StyleGAN (v1)**

![image](../../images/250408_10.PNG)

* How to run Fine-Tuning
  * **핵심 속성 값 (Property) 에 해당하는 size 7 의 Tensor** 를 Generator 의 입력 부분 및 Discriminator 의 Final Dense Layer 부분에 추가
  * Generator 와 Discriminator 의 **Conv. Layer 를 Freeze 시키고, Dense Layer 들만 추가 학습**
  * Generator Loss 가 Discriminator Loss 의 2배 이상이면, Discriminator 를 한번 학습할 때 **Generator 를 최대 4번까지 연속 학습** 하는 메커니즘 적용 
* Generator
  * ```stylegan_modified/stylegan_generator.py```
* Discriminator
  * ```stylegan_modified/stylegan_discriminator.py```
* Model Save Path
  * ```stylegan_modified/stylegan_gen_fine_tuned_v1.pth``` (**Generator** of **Modified Fine-Tuned** StyleGAN)
  * ```stylegan_modified/stylegan_dis_fine_tuned_v1.pth``` (**Discriminator** of **Modified Fine-Tuned** StyleGAN)

**3. Additional Fine-Tuned StyleGAN Generator (v2, CNN idea)**

![image](../../images/250408_11.PNG)

* How to run Fine-Tuning
  * 이미지로부터 Property Score 를 예측하는 Conv. NN (위 그림의 녹색 점선으로 표시한 부분) 을 먼저 학습
    * CNN 의 학습 데이터는 Original StyleGAN 으로 생성한 10,000 장 이미지 중 필터링된 여성 이미지 4,703 장 
  * 해당 CNN 을 Freeze 시킨 후, Fine-Tuned Generator (v1) 을 포함한 전체 신경망을 학습
* Generator
  * ```stylegan_modified/stylegan_generator_v2.py```
* Model Save Path
  * ```stylegan_modified/stylegan_gen_fine_tuned_v2.pth``` (Generator Model)
  * ```stylegan_modified/stylegan_gen_fine_tuned_v2_cnn.pth``` (**CNN** for Generator Model)

**4. Additional Fine-Tuned StyleGAN Generator (v3, Conditional VAE idea)**

![image](../../images/250408_12.PNG)

* How to run Fine-Tuning
  * Fine-Tuned Generator (v1) 을 **Conditional [VAE](https://github.com/WannaBeSuperteur/AI-study/blob/main/Generative%20AI/Basics_Variational%20Auto%20Encoder.md)** 의 Decoder 로 사용하여, Conditional VAE 를 학습
  * 필요에 따라 Conv. NN (Fine-Tuned Generator v2 에서 사용한) 을 Freeze 시켜서 사용할 수 있음
* Generator
  * ```stylegan_modified/stylegan_generator_v3.py```
* Model Save Path
  * ```stylegan_modified/stylegan_gen_fine_tuned_v3.pth``` (Generator Model)
  * ```stylegan_modified/stylegan_gen_fine_tuned_v3_encoder.pth``` (**Encoder of Conditional VAE** for Generator Model)

### 3-2. CNN Model

* CNN pipeline Overview

![image](../../images/250408_2.PNG)

* Training Data
  * ```cnn/synthesize_results_quality_and_gender.csv```
  * Dataset size
    * **2,000 rows**
    * each corresponding to first 2,000 images in ```stylegan/synthesize_results```
  * columns
    * **Image Quality** (0: Bad Quality, 1: Good Quality)
    * **Gender** (0: Man, 1: Woman)

* CNN Model Structure

![image](../../images/250408_4.PNG)

* Training Process
  * Loss Function 은 [Binary Cross-Entropy (BCE) Loss](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Loss_function.md#2-4-binary-cross-entropy-loss) 사용 [(MSE Loss 는 논리적으로 부적합)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Loss_Function_Misuse.md#1-1-probability-prediction-0--1-%EB%B2%94%EC%9C%84-%EB%8B%A8%EC%9D%BC-output-%EC%97%90%EC%84%9C-mse-loss-%EB%93%B1%EC%9D%B4-%EB%B6%80%EC%A0%81%EC%A0%88%ED%95%9C-%EC%9D%B4%EC%9C%A0)
  * 각 모델을 학습 시, **BCE Loss 가 일정 값 이상이면 학습 실패로 간주하여 재시도 (반복)** 하는 메커니즘 적용

| 핵심 속성 값              | 이미지의 학습 대상 영역                  | 학습 전략                                                                                                                                                                                                                                                          | 학습 전략 사용 이유                                                                                                                                                                                                                                                                                                             |
|----------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 성별 ```gender```      | 이미지 전체 (가로 256 x 세로 256)       | [K-fold Cross Validation](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%B0%A9%EB%B2%95%EB%A1%A0_Cross_Validation.md#3-k-fold-cross-validation)                       | - 라벨링된 데이터가 2,000 장으로 부족한 편                                                                                                                                                                                                                                                                                             |
| 이미지 품질 ```quality``` | 이미지 최상단 중앙 영역 (가로 128 x 세로 64) | [Startified K-fold Cross Validation](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%B0%A9%EB%B2%95%EB%A1%A0_Cross_Validation.md#4-stratified-k-fold-cross-validation) | - 라벨링된 데이터 부족<br>- **약 95% (1,905 / 2,000) 가 Good Quality 인 극단적인 [데이터 불균형](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4_%EA%B8%B0%EC%B4%88_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%B6%88%EA%B7%A0%ED%98%95.md)** |

* 핵심 속성 값 (성별, 이미지 품질) 데이터 저장 위치

| 이미지                     | 핵심 속성 값                    | 저장 위치                                               | repo 존재 여부<br>(clone 초기) |
|-------------------------|----------------------------|-----------------------------------------------------|--------------------------|
| 처음 2,000 장 (labeled)    | ```gender``` ```quality``` | ```cnn/synthesize_results_quality_and_gender.csv``` | **O**                    |
| 나머지 8,000 장 (unlabeled) | ```gender```               | ```cnn/inference_result/gender.csv```               | **O**                    |
| 나머지 8,000 장 (unlabeled) | ```quality```              | ```cnn/inference_result/quality.csv```              | **O**                    |
| **전체 이미지 10,000 장**     | ```gender``` ```quality``` | ```cnn/all_image_quality_and_gender.csv```          | **O**                    |

* CNN model 저장 경로
  * ```gender``` 모델 5개
    * ```cnn/models/gender_model_{0|1|2|3|4}.pt```
  * ```quality``` 모델 5개
    * ```cnn/models/quality_model_{0|1|2|3|4}.pt```

### 3-3. Segmentation Model (FaceXFormer)

[Implementation Source : FaceXFormer Official GitHub](https://github.com/Kartik-3004/facexformer/tree/main) (MIT License)

* Main Model Save Path ([Original Source](https://huggingface.co/kartiknarayan/facexformer/tree/main/ckpts))
  * ```segmentation/models/segmentation_model.pt``` (Pre-trained FaceXFormer)
* Additional Models Save Path ([Original Source](https://github.com/timesler/facenet-pytorch/blob/master/data))  
  * ```segmentation/models/mtcnn_pnet.pt``` (Pre-trained P-Net for MTCNN)
  * ```segmentation/models/mtcnn_rnet.pt``` (Pre-trained R-Net for MTCNN)
  * ```segmentation/models/mtcnn_onet.pt``` (Pre-trained O-Net for MTCNN)

## 4. 코드 실행 방법

**모든 코드는 아래 순서대로, ```2025_04_08_OhLoRA``` main directory 에서 실행** (단, 추가 개발 목적이 아닌 경우, 마지막의 **"6. Fine-Tuning 된 StyleGAN 실행하여 이미지 생성"** 부분만 실행)

* **1. Original GAN Generator 실행하여 이미지 생성**
  * ```python stylegan_and_segmentation/run_original_generator.py```
  * ```stylegan/synthesize_results``` 에 생성된 이미지 저장됨

* **2. CNN 실행**
  * ```python stylegan_and_segmentation/run_cnn.py```
  * 모든 이미지에 대한 핵심 속성 값 데이터 (unlabeled image 의 경우 모델 계산값) 가 저장됨
  * CNN model 이 지정된 경로에 없을 시, CNN 모델 학습
  * ```stylegan/synthesize_results_filtered``` 에 필터링된 이미지 저장됨 **(StyleGAN Fine-Tuning 학습 데이터로 사용)**

* **3. Segmentation 결과 생성**
  * 전체 10,000 장이 아닌, 그 일부분에 해당하는 **따로 필터링된 이미지 4,703 장** 대상 
  * ```python stylegan_and_segmentation/run_segmentation.py```
  * ```segmentation/segmentation_results``` 에 이미지 저장됨

* **4. 성별, 이미지 품질을 제외한 7가지 핵심 속성값 계산 결과 생성**
  * 전체 10,000 장이 아닌, 그 일부분에 해당하는 **따로 필터링된 이미지 4,703 장** 대상 
  * ```python stylegan_and_segmentation/compute_property_scores.py```
  * ```segmentation/property_score_results``` 에 결과 저장됨

* **5. StyleGAN Fine-Tuning 실시**
  * 전체 10,000 장이 아닌, 그 일부분에 해당하는 **따로 필터링된 이미지 4,703 장** 대상 
  * **StyleGAN-FineTune-v1** 
    * ```python stylegan_and_segmentation/run_stylegan_fine_tuning.py```
    * ```stylegan_modified/stylegan_gen_fine_tuned.pth``` 에 Fine-Tuning 된 모델의 Generator 저장됨
    * ```stylegan_modified/stylegan_dis_fine_tuned.pth``` 에 Fine-Tuning 된 모델의 Discriminator 저장됨
    * 위 2개의 모델은 이름을 ```... fine_tuned.pth``` 에서 ```... fine_tuned_v1.pth``` 로 각각 변경하여 사용
  * **StyleGAN-FineTune-v2 (CNN 기반)** 
    * ```python stylegan_and_segmentation/run_stylegan_fine_tuning_v2.py```
    * ```stylegan_modified/stylegan_gen_fine_tuned_v2.pth``` 에 Fine-Tuning 된 Generator 저장됨
  * **StyleGAN-FineTune-v3 (Conditional VAE 기반)** 
    * ```python stylegan_and_segmentation/run_stylegan_fine_tuning_v3.py```
    * ```stylegan_modified/stylegan_gen_fine_tuned_v3.pth``` 에 Fine-Tuning 된 Generator 저장됨

* **6. Fine-Tuning 된 StyleGAN 실행하여 이미지 생성**
  * ```python stylegan_and_segmentation/run_fine_tuned_generator.py```
  * ```stylegan_modified/synthesize_results``` 에 생성된 이미지 저장됨
