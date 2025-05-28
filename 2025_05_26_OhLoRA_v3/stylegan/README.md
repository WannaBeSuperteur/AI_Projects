## 1. 개요

* 핵심 요약
  * **Oh-LoRA 👱‍♀️ (오로라) 프로젝트의 v3 버전** 에서 사용하는 **가상 인간 여성 이미지 생성 알고리즘**
* 모델 구조 요약
  * Original StyleGAN
  * → StyleGAN-FineTune-v1 **('속성 값' 으로 conditional 한 이미지 생성 시도)**
  * → StyleGAN-FineTune-v8 **(Oh-LoRA 컨셉에 맞는 이미지로 추가 Fine-Tuning)** 
  * → StyleGAN-VectorFind-v8 **(Oh-LoRA 의 표정을 변화시키는 intermediate 'w vector' 를 활용)**

### 1-1. 모델 구조

![image](../../images/250526_4.PNG)

* Original StyleGAN (출처 : [GenForce GitHub](https://github.com/genforce/genforce/blob/master/MODEL_ZOO.md) > StyleGAN Ours > **celeba-partial-256x256**) → StyleGAN-FineTune-v1 [(참고)](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan)
  * Original StyleGAN 으로 10,000 장의 이미지 생성
  * 그 중 **고품질 여성 이미지** 4,703 장을 필터링
  * [핵심 속성 값](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#2-핵심-속성-값) (```eyes``` ```hair_color``` ```hair_length``` ```mouth``` ```pose``` ```background_mean``` ```background_std```) 계산을 위한 [Property Score CNN](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#3-3-cnn-model-나머지-핵심-속성-값-7개) 학습

* StyleGAN-FineTune-v1 → StyleGAN-FineTune-v8
  * StyleGAN-FineTune-v1 으로 15,000 장의 이미지 생성
  * 그 중 **안경을 쓰지 않은 고품질의 젊은 여성 이미지** 4,930 장을 필터링
  * ```Hairstyle``` (직모 vs. 곱슬머리) 속성 값 계산을 위한 CNN 학습

* StyleGAN-VectorFind-v8
  * StyleGAN 의 w vector 에 더하거나 뺌으로서 ```eyes``` ```mouth``` ```pose``` 핵심 속성 값을 **가장 잘 변화시키는** vector 를 탐색
  * [참고 논문](https://arxiv.org/pdf/1911.09267) 및 [스터디 자료](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)
  * [StyleGAN 에서, mapping 이전의 z vector 보다는 **mapping 이후의 w vector** 가 핵심 속성 값을 잘 변화시키는 vector 탐색에 좋음](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.04.09%5D%20A%20Style-Based%20Generator%20Architecture%20for%20Generative%20Adversarial%20Networks.md#4-1-feature-%EB%A1%9C%EC%9D%98-mapping-%EB%B9%84%EA%B5%90)
    * 'z vector' 보다는 'intermediate w vector' 가 **덜 entangle 되어 있음**
    * **entangle** 이란, 한 속성이 변화하면 다른 속성도 변하는 (예: 눈을 작게 뜨면 입이 벌어지는) 현상을 의미함

## 2. 핵심 속성 값

* **핵심 속성 값** 은 다음 목적을 위해 사용되는 numeric value 를 말한다.
  * Oh-LoRA 👱‍♀️ (오로라) 얼굴 생성을 위한 고품질의 이미지 필터링용 조건 값
  * 조건에 맞는 Oh-LoRA 👱‍♀️ (오로라) 얼굴 이미지 생성을 위한 조건 값
  * 이미지 그룹화 등 AI 학습 성능을 높이는 목적으로도 사용
* 아래 표에서 $N(0, 1)$ 은 다음을 의미한다.
  * **[알고리즘](#2-1-핵심-속성-값-계산-알고리즘) 에 의해 계산된 원래 핵심 속성 값** 이, Normal Distribution (표준정규분포) 으로 **먼저 정규화된 후에 실제 모델에 입력됨**

| 핵심 속성 값 이름                   | 설명                                    | 값 범위 or 분포  | 이미지 필터링에 사용 | 용도                                                                                          |
|------------------------------|---------------------------------------|-------------|-------------|---------------------------------------------------------------------------------------------|
| 성별 ```gender```              | 0 (남성) ~ 1 (여성) 의 확률 값                | 0 ~ 1       | ✅           | 이미지 필터링<br>(**1 에 가까워야** 합격)                                                                |
| 이미지 품질 ```quality```         | 0 (저품질) ~ 1 (고품질) 의 확률 값              | 0 ~ 1       | ✅           | 이미지 필터링<br>(**1 에 가까워야** 합격)                                                                |
| 나이 ```age```                 | 0 (젊음) ~ 1 (나이 듦) 의 확률 값              | 0 ~ 1       | ✅           | 이미지 필터링<br>(**0 에 가까워야** 합격)                                                                |
| 안경 여부 ```glass```            | 0 (안경 X) ~ 1 (안경 O) 의 확률 값            | 0 ~ 1       | ✅           | 이미지 필터링<br>(**0 에 가까워야** 합격)                                                                |
| 눈을 뜬 정도 ```eyes```           | 눈을 크게 뜰수록 값이 큼                        | $N(0, 1^2)$ | ❌           | Oh-LoRA 👱‍♀️ (오로라) 의 **표정 제어**                                                             |
| 머리 색 ```hair_color```        | 머리 색이 밝을수록 값이 큼                       | $N(0, 1^2)$ | ❌           | Oh-LoRA 👱‍♀️ (오로라) 의 표정을 가장 잘 제어하는 w vector 탐색을 위해서는 SVM 필요.<br>이 SVM 의 학습을 위한 **이미지 그룹화** |
| 머리 길이 ```hair_length```      | 머리 길이가 길수록 값이 큼                       | $N(0, 1^2)$ | ❌           | Oh-LoRA 👱‍♀️ (오로라) 의 표정 제어 w vector 탐색용 SVM 의 학습을 위한 **이미지 그룹화**                           |
| 입을 벌린 정도 ```mouth```         | 입을 벌린 정도가 클수록 값이 큼                    | $N(0, 1^2)$ | ❌           | Oh-LoRA 👱‍♀️ (오로라) 의 **표정 제어**                                                             |
| 고개 돌림 ```pose```             | 왼쪽 고개 돌림 (-1), 정면 (0), 오른쪽 고개 돌림 (+1) | $N(0, 1^2)$ | ❌           | Oh-LoRA 👱‍♀️ (오로라) 의 **표정 제어**                                                             |
| 배경색 평균 ```background_mean``` | 이미지 배경 부분 픽셀의 색의 평균값이 클수록 값이 큼        | $N(0, 1^2)$ | ❌           | Oh-LoRA 👱‍♀️ (오로라) 의 표정 제어 w vector 탐색용 SVM 의 학습을 위한 **이미지 그룹화**                           |
| 직모 vs. 곱슬머리 ```hairstyle```  | 직모보다 곱슬머리에 가까울수록 값이 큼                 | $N(0, 1^2)$ | ❌           | Oh-LoRA 👱‍♀️ (오로라) 의 표정 제어 w vector 탐색용 SVM 의 학습을 위한 **이미지 그룹화**                           |

* [참고 (오로라 v1 프로젝트 문서)](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#2-핵심-속성-값)

### 2-1. 핵심 속성 값 계산 알고리즘

| 핵심 속성 값 이름                                                                                                                   | 계산 알고리즘                                                                                                                                               |
|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| - 성별 ```gender```<br>- 이미지 품질 ```quality```<br>- 나이 ```age```<br>- 안경 여부 ```glass```                                         |                                                                                                                                                       |
| - 눈을 뜬 정도 ```eyes```<br>- 머리 색 ```hair_color```<br>- 머리 길이 ```hair_length```<br>- 입을 벌린 정도 ```mouth```<br>- 고개 돌림 ```pose``` | [계산 알고리즘 설명 (오로라 v1 프로젝트 문서)](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#2-2-핵심-속성-값-계산-알고리즘-2차-알고리즘-for-stylegan-finetune-v2-v3-v4) |
| - 배경색 평균 ```background_mean```                                                                                               | [계산 알고리즘 설명 (오로라 v1 프로젝트 문서)](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#2-1-핵심-속성-값-계산-알고리즘-1차-알고리즘-for-stylegan-finetune-v1)       |
| - 직모 vs. 곱슬머리 ```hairstyle```                                                                                                | [계산 알고리즘 설명](#2-2-직모-vs-곱슬머리-속성-값-계산-알고리즘)                                                                                                            |

### 2-2. 직모 vs. 곱슬머리 속성 값 계산 알고리즘

## 코드 실행 방법

모든 코드는 ```2025_05_26_OhLoRA_v3``` (프로젝트 메인 디렉토리) 에서 실행

* **StyleGAN-FineTune-v8** 의 Fine-Tuning 에 필요한 15,000 장의 사람 얼굴 이미지 생성
  * ```python stylegan/run_generate_dataset.py``` 