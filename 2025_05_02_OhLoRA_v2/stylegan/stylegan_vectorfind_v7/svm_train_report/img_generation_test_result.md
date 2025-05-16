## 목차

* [1. Final Report](#1-final-report)
* [2. Grouping](#2-grouping)
* [3. Image Generation Test Result](#3-image-generation-test-result)

## 1. Final Report

* random z latent vector 를 **아래 결과처럼 100 개가 아닌, 이보다 훨씬 많은 개수** 로 하여 테스트
* [grouping](#2-grouping) (8 groups) 는 **모든 case 에 대해 항상 적용**
* 결론
  * **Grouping 적용 시 유의미하게 성능이 향상됨 (크게 향상되지는 않음)**
* passed 기준 **(모두 만족)**
  * 각 속성 값 별, **의도한 값 vs. 실제 생성된 이미지에 대해 Property Score CNN 으로 도출한 값** 의 corr-coef (상관계수) 가 다음을 만족 
  * ```eyes``` : 상관계수의 절댓값이 **0.92 이상** ([v6](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) : 0.75 이상)
  * ```mouth``` : 상관계수의 절댓값이 **0.88 이상** ([v6](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) : 0.77 이상)
  * ```pose``` : 상관계수의 절댓값이 **0.92 이상** ([v6](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) : 0.80 이상)

| n<br>(total samples) | k<br>(top / bottom samples) | latent vectors<br>(random z) | passed cases | Final Oh-LoRA 적합 case | ```eyes``` mean corr-coef | ```mouth``` mean corr-coef | ```pose``` mean corr-coef | details<br>(csv) |
|----------------------|-----------------------------|------------------------------|--------------|-----------------------|---------------------------|----------------------------|---------------------------|------------------|
|                      |                             |                              |              |                       |                           |                            |                           |                  |

## 2. Grouping

* 이미지를 random latent code (z) 로부터 생성할 때,
  * 해당 이미지의 **머리 색, 머리 길이, 배경 색 평균** 의 핵심 속성 값을 Property Score CNN 으로 예측
  * 해당 핵심 속성 값을 기준으로 이미지를 각 그룹으로 분류
* 각 그룹별로,
  * t-SNE 실시 및 SVM 학습
  * **눈을 뜬 정도, 입을 벌린 정도, 고개 돌림** 각각에 대한 n vector 계산
* inference 시,
  * 생성된 이미지가 어느 그룹에 속하는지 Property Score CNN 으로 판단
  * 해당 그룹에 맞는 **눈을 뜬 정도, 입을 벌린 정도, 고개 돌림** n vector 를 이용하여 핵심 속성 값이 변동된 이미지 생성

## 3. Image Generation Test Result

* **CONCLUSION**
  * **StyleGAN-VectorFind-v7 is MUCH BETTER than StyleGAN-VectorFind-v6**

* experiment settings
  * with both **[sklearnex](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)** and **[grouping](#2-grouping) (8 groups)** applied for all cases
  * always used ```LinearSVC(...)``` instead of ```SVC(kernel='linear', ...)```

* how to analyze table
  * performance (accuracy of ```eyes```, ```mouth``` and ```pose```) means **SVM accuracy**
  * for **mean corr-coef**,
    * each corr-coef means the corr-coef of **Intended Property Scores vs. Actual CNN-Predicted Property Scores** for 50 generated images
      * **Note: Intended Property Scores** are **different** from those of **StyleGAN-VectorFind-v6**.
    * total 100 cases (random z latent vectors) for each experiment

* comparison
  * vs. **StyleGAN-VectorFind-v6**

| n<br>(total samples) | k<br>(top / bottom samples) | ```eyes``` acc.<br>(0 ~ 1)                                                                           | ```mouth``` acc.<br>(0 ~ 1)                                                                          | ```pose``` acc.<br>(0 ~ 1)                                                                           | ```eyes``` mean corr-coef                                                                            | ```mouth``` mean corr-coef                                                                           | ```pose``` mean corr-coef                                                                            |
|----------------------|-----------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 4.0K                 | 800 / 800                   | 0.9565<br>[(🔺 0.3430)](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) | 0.9317<br>[(🔺 0.2223)](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) | 0.9224<br>[(🔺 0.1568)](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) | 0.8808<br>[(🔺 0.1460)](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) | 0.8522<br>[(🔺 0.2255)](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) | 0.8433<br>[(🔺 0.2823)](../../stylegan_vectorfind_v6/svm_train_report/img_generation_test_result.md) |
| 10.0K                | 2.0K / 2.0K                 |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |
| 20.0K                | 4.0K / 4.0K                 |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |
| 50.0K                | 10.0K / 10.0K               |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |
| 100.0K               | 15.0K / 15.0K               |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |                                                                                                      |
