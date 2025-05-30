## 개요

* StyleGAN-VectorFind-v8 학습 시 **이미지를 ```hair_color``` ```hair_length``` ```background_mean``` ```hairstyle``` 속성 값에 따라 MBTI 처럼 $2^4 = 16$ 개 그룹으로 나누어 SVM 학습**
* 각 속성 값 별 구분 기준점은 StyleGAN-FineTune-v8 로 생성한 이미지들에 대해 **각 속성 값의 median (중앙값)** 을 적용
  * 구분 기준점 비유 : MBTI 의 E vs. I 를 나누는 외향적 성향의 정도
* 본 디렉토리는 해당 중앙값을 구하기 위한 **StyleGAN-FineTune-v8 생성 이미지에 대한 해당 속성 값 데이터** 를 저장

## 코드 실행 방법

모든 코드는 ```2025_05_26_OhLoRA_v3``` (프로젝트 메인 디렉토리) 에서 실행

* **StyleGAN-FineTune-v8** 으로 이미지 생성 및 생성된 이미지에 대한 ```hair_color``` ```hair_length``` ```background_mean``` ```hairstyle``` 속성 값 데이터 저장
  * ```python v8_property_scores/run_generate_and_evaluate_images.py``` 