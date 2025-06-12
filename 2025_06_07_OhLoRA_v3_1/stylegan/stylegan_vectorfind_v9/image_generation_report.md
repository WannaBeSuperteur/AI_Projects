## 목차

* [1. Final Report](#1-final-report)
* [2. Image Generation Test Result](#2-image-generation-test-result)

## 1. Final Report

* 실험 설정
  * ```SVM``` 및 ```Gradient``` 방법론으로 실험 실시

| 방법론            | 핵심 아이디어                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 상세 정보 |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| ```SVM```      | - [StyleGAN-VectorFind-v8](../../../2025_05_26_OhLoRA_v3/stylegan/README.md#3-3-stylegan-finetune-v8-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v8) 과 같이, [SVM (Support Vector Machine)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%AA%A8%EB%8D%B8_SVM.md) 으로 핵심 속성 값 변화 벡터 도출<br>- [grouping](../../../2025_05_26_OhLoRA_v3/stylegan/stylegan_vectorfind_v8/image_generation_report.md#2-grouping) (16 groups) 은 **모든 case 에 대해 항상 적용** |       |
| ```Gradient``` | - 간단한 딥러닝 모델의 Gradient 를 핵심 속성 값 변화 벡터로 사용                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |       |

* passed 기준 **(모두 만족)**
  * 각 속성 값 별, **의도한 값 vs. 실제 생성된 이미지에 대해 Property Score CNN 으로 도출한 값** 의 corr-coef (상관계수) 가 다음을 만족 
  * ```eyes``` : 상관계수의 절댓값이 **(TBU) 이상** ([v7](../../../2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v7/svm_train_report/img_generation_test_result.md) : 0.92 이상)
  * ```mouth``` : 상관계수의 절댓값이 **(TBU) 이상** ([v7](../../../2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v7/svm_train_report/img_generation_test_result.md) : 0.88 이상)
  * ```pose``` : 상관계수의 절댓값이 **(TBU) 이상** ([v7](../../../2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v7/svm_train_report/img_generation_test_result.md) : 0.92 이상)

* 최종 이미지 생성 결과
  * 각 핵심 속성 값 별 다음 방법 적용 
    * ```eyes``` 속성값 : ```Gradient``` 방법 적용
    * ```mouth``` ```pose``` 속성값 : ```SVM``` 방법 적용

| n<br>(total samples) | k<br>(top / bottom samples) | latent vectors<br>(random z) | passed cases | Final Oh-LoRA 적합 case | ```eyes``` mean corr-coef | ```mouth``` mean corr-coef | ```pose``` mean corr-coef | details<br>(csv) |
|----------------------|-----------------------------|------------------------------|--------------|-----------------------|---------------------------|----------------------------|---------------------------|------------------|
|                      | / <br>**(% each)**          |                              | ( %)         | ( %)                  |                           |                            |                           |                  |

## 2. Image Generation Test Result

* **결론**
  * ```eyes``` 속성값은 ```Gradient``` 방법론, 나머지 ```mouth``` ```pose``` 속성값은 ```SVM``` 방법론의 성능이 우수하다. 
  * ```Gradient``` 방법론에 한해서, 모델 학습을 위해 생성하는 후보 이미지의 개수 (= random latent code z 의 개수), 즉 **Sample Count (N)** 이 커질수록 **성능 (mean corr-coef 기준)** 이 좋다.

* 실험 설정
  * ```SVM``` 방법론의 경우, 모든 실험에서 **[sklearnex](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)** 및 **[grouping](#2-grouping) (16 groups)** 둘 다 적용
  * ```SVM``` 방법론의 경우, 모든 실험에서 ```SVC(kernel='linear', ...)``` 대신 **```LinearSVC(...)``` 를 적용**

* 그래프 해석
  * mean corr-coef
    * 각 corr-coef (상관계수) 는 **각 case 별 50 장의 생성된 이미지** 에 대해, **의도한 Property Scores vs. 실제 CNN 에 의해 도출된 Property Scores** 의 상관계수
    * 각 실험 별 100 cases (random z latent vectors)
  * 범례
    * 범례는 ```{방법론 이름} / {intermediate vector 추출 방법 이름}``` 형식
    * intermediate vector 추출 방법은 해당 문서 (TBU) 참고

* **3개 핵심 속성 값 (```eyes``` ```mouth``` ```pose```) 의 mean corr-coef 합계**

![image](../../../images/250607_15.PNG)

### 2-1. 각 핵심 속성 값 별 상세 결과

* 전체 요약
  * ```Gradient``` 방법론에서 일부 경우에서 **Gradient 추출용 딥러닝 모델의 학습 데이터가 늘어남에 따라 성능이 감소하는 이유는 불명**
  * 해당 딥러닝 모델 자체의 [MSE Error](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Loss_function.md#2-1-mean-squared-error-mse) 는 **학습 데이터가 늘어남에 따라 감소하는 추세** 이므로, **딥러닝 학습은 문제 없음** 

| 핵심 속성 값     | 우수한 방법론        | intermediate vector 추출 레이어 별 성능 차이    | ```SVM``` 방법론 결과 특징             | ```Gradient``` 방법론 결과 특징                                                                                                                   |
|-------------|----------------|---------------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| ```eyes```  | ```Gradient``` | ```SVM```, ```Gradient``` 모두 유의미하지 않음 | 생성 이미지 개수 (n) 이 **증가할수록 성능 향상** | 생성 이미지 개수 (n) 에 따른 성능 변화 거의 없음                                                                                                             |
| ```mouth``` | ```SVM```      | ```SVM```, ```Gradient``` 모두 유의미하지 않음 | 생성 이미지 개수 (n) 이 **증가할수록 성능 향상** | 생성 이미지 개수 (n) 이 **증가할수록 성능 소폭 하향 추정**                                                                                                      |
| ```pose```  | ```SVM```      | ```SVM```, ```Gradient``` 모두 유의미하지 않음 | 생성 이미지 개수 (n) 에 따른 성능 변화 거의 없음  | - ```mapping_split2``` 레이어에서 intermediate vector 추출 시, 생성 이미지 개수 (n) 이 **증가할수록 성능 하향**<br>- 나머지 레이어에서 추출 시, 생성 이미지 개수 (n) 에 따른 성능 변화 거의 없음 |

* ```eyes``` 핵심 속성 값의 mean corr-coef 

![image](../../../images/250607_12.PNG)

* ```mouth``` 핵심 속성 값의 mean corr-coef

![image](../../../images/250607_13.PNG)

* ```pose``` 핵심 속성 값의 mean corr-coef

![image](../../../images/250607_14.PNG)

* csv 파일

| intermediate vector 추출 레이어 | ```SVM``` 방법론                                              | ```Gradient``` 방법론                                              |
|----------------------------|------------------------------------------------------------|-----------------------------------------------------------------|
| ```mapping_split1```       | [csv file](test_result/test_result_svm_mapping_split1.csv) | [csv file](test_result/test_result_gradient_mapping_split1.csv) |
| ```mapping_split2```       | [csv file](test_result/test_result_svm_mapping_split2.csv) | [csv file](test_result/test_result_gradient_mapping_split2.csv) |
| ```w```                    | [csv file](test_result/test_result_svm_w.csv)              | [csv file](test_result/test_result_gradient_w.csv)              |

