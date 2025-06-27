
## StyleGAN-VectorFind 모델 정보

| 모델                     | Oh-LoRA 버전                                                   | 핵심 아이디어                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| StyleGAN-VectorFind-v7 | [Oh-LoRA v2 ('25.05.02 - 05.21)](../../2025_05_02_OhLoRA_v2) | - **핵심 속성값을 잘 변화** 시키는, latent vector z 에 대한 **벡터 찾기** [(논문 스터디 자료)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Paper%20Study/Vision%20Model/%5B2025.05.05%5D%20Semantic%20Hierarchy%20Emerges%20in%20Deep%20Generative%20Representations%20for%20Scene%20Synthesis.md)<br>- latent vector z 대신 **entangle (속성 간 얽힘) 이 보다 덜한 intermediate vector w 를 이용**<br>- 이미지 특징에 따른 정밀한 SVM 학습을 위해, ```머리 색``` ```머리 길이``` ```배경 밝기``` 를 기준으로 총 $8 = 2^3$ 그룹으로 나누어, 각 그룹별 SVM 학습 |
| StyleGAN-VectorFind-v8 | [Oh-LoRA v3 ('25.05.26 - 06.05)](../../2025_05_26_OhLoRA_v3) | - **StyleGAN-VectorFind-v7** 기반<br>- ```안경을 쓰지 않은 젊은 여성을 나타내는 고품질의 이미지``` 만을 필터링한 StyleGAN-FineTune-v8 로 학습<br>- SVM 학습을 위한 그룹 분류 기준에 ```직모/곱슬``` 을 추가하여, 총 $16 = 2^4$ 그룹에 대해 각 그룹별 SVM 학습                                                                                                                                                                                                                                                                                         |

* [StyleGAN-VectorFind-v7 설명](../../2025_05_02_OhLoRA_v2/stylegan/README.md#3-3-stylegan-finetune-v1-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v7)

![image](../../images/250502_23.PNG)

* [StyleGAN-VectorFind-v8 설명](../../2025_05_26_OhLoRA_v3/stylegan/README.md#3-3-stylegan-finetune-v8-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v8)

![image](../../images/250526_12.png)

* [StyleGAN-FineTune-v8 설명](../../2025_05_26_OhLoRA_v3/stylegan/README.md#3-2-fine-tuned-stylegan-stylegan-finetune-v8)

![image](../../images/250526_3.PNG)

## 코드 실행 방법

모든 코드는 ```2025_06_24_OhLoRA_v4``` (프로젝트 메인 디렉토리) 에서 실행 **(단, 먼저 HuggingFace Link (TBU) 에서 모델 파일 다운로드 후, 모델 파일 경로 정보 (TBU) 에 따라 해당 파일들을 알맞은 경로에 배치)**

* **StyleGAN-VectorFind-v7** 모델 실행 (Oh-LoRA 👱‍♀️ 얼굴 이미지 생성)
  * ```python stylegan/run_stylegan_vectorfind_v7.py```

* **StyleGAN-VectorFind-v8** 모델 실행 (Oh-LoRA 👱‍♀️ 얼굴 이미지 생성)
  * ```python stylegan/run_stylegan_vectorfind_v8.py```