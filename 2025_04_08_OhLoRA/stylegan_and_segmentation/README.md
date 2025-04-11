## 목차

* [1. 개요](#1-개요)
* [2. 핵심 속성 값](#2-핵심-속성-값)
  * [2-1. 핵심 속성 값 계산 알고리즘](#2-1-핵심-속성-값-계산-알고리즘)
* [3. 사용 모델 설명](#3-사용-모델-설명)
  * [3-1. Image Generation Model (StyleGAN)](#3-1-image-generation-model-stylegan)
  * [3-2. Segmentation Model (FaceXFormer)](#3-2-segmentation-model-facexformer)
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
| 머리 길이 ```hair_length```        | 머리 길이가 길수록 값이 큼                   | 0 ~ 1   |
| 입을 벌린 정도 ```mouth```           | 입을 벌린 정도가 클수록 값이 큼                | 0 ~ 1   |
| 얼굴의 위치 ```face_pose```         | 왼쪽 쳐다봄 (-1), 정면 (0), 오른쪽 쳐다봄 (+1) | -1 ~ +1 |

### 2-1. 핵심 속성 값 계산 알고리즘

* Segmentation 결과를 바탕으로 다음과 같이 **성별, 이미지 품질을 제외한 5가지 핵심 속성 값들을 계산**
  * 계산 대상 핵심 속성 값 
    * 눈을 뜬 정도, 머리 색, 머리 길이, 입을 벌린 정도, 얼굴의 위치 
  * 점수 계산 완료 후, **모든 이미지에 대해 각 속성 종류별로 그 값들을 위 표의 "값 범위"에 따라 [min-max normalization](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4_%EA%B8%B0%EC%B4%88_Normalization.md#2-1-min-max-normalization) 적용**
    * 예를 들어, 모든 이미지에 대한 머리 색의 값이 ```[100, 250, 120, 180, 210]``` 인 경우, 이를 linear transform 하여 ```[0.0, 1.0, 0.133, 0.533, 0.733]``` 으로 정규화
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

## 3. 사용 모델 설명

### 3-1. Image Generation Model (StyleGAN)

[Implementation & Pre-trained Model Source : GenForce GitHub](https://github.com/genforce/genforce/tree/master) (MIT License)

* Generator
  * ```stylegan/stylegan_generator.py```
* Discriminator
  * ```stylegan/stylegan_discriminator.py```
* Model Save Path
  * ```stylegan/stylegan_model.pth``` (Original GAN, including Generator & Discriminator)
    * from [MODEL ZOO](https://github.com/genforce/genforce/blob/master/MODEL_ZOO.md) > StyleGAN Ours > **celeba-partial-256x256**

### 3-2. Segmentation Model (FaceXFormer)

[Implementation Source : FaceXFormer Official GitHub](https://github.com/Kartik-3004/facexformer/tree/main) (MIT License)

* Main Model Save Path ([Original Source](https://huggingface.co/kartiknarayan/facexformer/tree/main/ckpts))
  * ```segmentation/models/segmentation_model.pt``` (Pre-trained FaceXFormer)
* Additional Models Save Path ([Original Source](https://github.com/timesler/facenet-pytorch/blob/master/data))  
  * ```segmentation/models/mtcnn_pnet.pt``` (Pre-trained P-Net for MTCNN)
  * ```segmentation/models/mtcnn_rnet.pt``` (Pre-trained R-Net for MTCNN)
  * ```segmentation/models/mtcnn_onet.pt``` (Pre-trained O-Net for MTCNN)

## 4. 코드 실행 방법

**모든 코드는 아래 순서대로, ```2025_04_08_OhLoRA``` main directory 에서 실행**

* Original GAN Generator 실행하여 이미지 생성
  * ```python stylegan_and_segmentation/run_original_generator.py```
  * ```stylegan/synthesize_results``` 에 생성된 이미지 저장됨

* Segmentation 결과 생성
  * 전체 10,000 장이 아닌, 그 일부분에 해당하는 **따로 필터링된 이미지** 대상 
  * ```python stylegan_and_segmentation/run_segmentation.py```
  * ```segmentation/segmentation_results``` 에 이미지 저장됨

* 성별, 이미지 품질을 제외한 5가지 핵심 속성값 계산 결과 생성
  * 전체 10,000 장이 아닌, 그 일부분에 해당하는 **따로 필터링된 이미지** 대상 
  * ```python stylegan_and_segmentation/compute_property_scores.py```
  * ```segmentation/property_score_results``` 에 결과 저장됨