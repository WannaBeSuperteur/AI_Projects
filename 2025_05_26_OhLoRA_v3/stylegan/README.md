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
  * [StyleGAN 에서, mapping 이전의 z vector 보다는 **mapping 이후의 w vector** 가 핵심 속성 값을 잘 변화시키는 vector 탐색에 좋음](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.04.09%5D%20A%20Style-Based%20Generator%20Architecture%20for%20Generative%20Adversarial%20Networks.md)

## 코드 실행 방법

모든 코드는 ```2025_05_26_OhLoRA_v3``` (프로젝트 메인 디렉토리) 에서 실행

* **StyleGAN-FineTune-v8** 의 Fine-Tuning 에 필요한 15,000 장의 사람 얼굴 이미지 생성
  * ```python stylegan/run_generate_dataset.py``` 