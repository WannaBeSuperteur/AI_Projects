## 목차

* [1. Final Report](#1-final-report)
* [2. Grouping](#2-grouping)
* [3. Image Generation Test Result](#3-image-generation-test-result)

## 1. Final Report

* random z latent vector 를 **아래 결과처럼 100 개가 아닌, 이보다 훨씬 많은 개수** 로 하여 테스트
* 결론
  * **Grouping 적용 시 유의미하게 성능이 향상됨 (크게 향상되지는 않음)**
* passed 기준 **(모두 만족)**
  * 각 속성 값 별, **의도한 값 vs. 실제 생성된 이미지에 대해 Property Score CNN 으로 도출한 값** 의 corr-coef (상관계수) 가 다음을 만족 
  * ```eyes``` : 상관계수의 절댓값이 **0.75 이상**
  * ```mouth``` : 상관계수의 절댓값이 **0.77 이상**
  * ```pose``` : 상관계수의 절댓값이 **0.8 이상**

| n<br>(total samples) | k<br>(top / bottom samples) | [grouping](#2-grouping)<br>(8 groups) | latent vectors<br>(random z) | passed cases  | Final Oh-LoRA 적합 case | ```eyes``` mean corr-coef | ```mouth``` mean corr-coef | ```pose``` mean corr-coef | details<br>(csv)                                                                                                                                                                                                                                                                                                                                                                                                |
|----------------------|-----------------------------|---------------------------------------|------------------------------|---------------|-----------------------|---------------------------|----------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 300.0K               | 30.0K / 30.0K               | ❌                                     | 763                          | 12 (1.6%)     | 3 (0.4%)              | **0.7637**                | **0.7003**                 | 0.5616                    | [summary](https://github.com/WannaBeSuperteur/AI_Projects/blob/3a1cae36a05f195d63776f31929756906fb6d70a/2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v6/image_generation_report/test_statistics.csv), [detail](https://github.com/WannaBeSuperteur/AI_Projects/blob/3a1cae36a05f195d63776f31929756906fb6d70a/2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v6/image_generation_report/test_result.csv) |
| 500.0K               | 75.0K / 75.0K               | ✅                                     | 475                          | **25 (5.3%)** | **9 (1.9%)**          | 0.7348                    | 0.6995                     | **0.6686**                | [summary](https://github.com/WannaBeSuperteur/AI_Projects/blob/c35570cd2a942b52d33743f18bea4c68473d0c68/2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v6/image_generation_report/test_statistics.csv), [detail](https://github.com/WannaBeSuperteur/AI_Projects/blob/c35570cd2a942b52d33743f18bea4c68473d0c68/2025_05_02_OhLoRA_v2/stylegan/stylegan_vectorfind_v6/image_generation_report/test_result.csv) |

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

* with **[sklearnex](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)** for all cases
* always used ```LinearSVC(...)``` instead of ```SVC(kernel='linear', ...)```
* performance (accuracy of ```eyes```, ```mouth``` and ```pose```) means **SVM accuracy**
* for **mean corr-coef**,
  * each corr-coef means the corr-coef of **Intended Property Scores vs. Actual CNN-Predicted Property Scores** for 50 generated images
  * total 100 cases (random z latent vectors) for each experiment

| n<br>(total samples) | k<br>(top / bottom samples) | [grouping](#2-grouping)<br>(8 groups) | ```eyes``` acc.<br>(0 ~ 1) | ```mouth``` acc.<br>(0 ~ 1) | ```pose``` acc.<br>(0 ~ 1) | ```eyes``` mean corr-coef | ```mouth``` mean corr-coef | ```pose``` mean corr-coef |
|----------------------|-----------------------------|---------------------------------------|----------------------------|-----------------------------|----------------------------|---------------------------|----------------------------|---------------------------|
| 4.0K                 | 800 / 800                   | ❌                                     | 0.7000                     | 0.7094                      | 0.7656                     | 0.7348                    | 0.6267                     | 0.5610                    |
| 4.0K                 | 800 / 800                   | ✅                                     | 0.6135                     | 0.6472                      | 0.6902                     | 0.6744                    | 0.5504                     | 0.5963                    |
| 50.0K                | 800 / 800                   | ❌                                     | 0.8156                     | 0.8187                      | **0.9219**                 | 0.7081                    | 0.6971                     | 0.5206                    |
| 50.0K                | 4.0K / 4.0K                 | ❌                                     | 0.8525                     | **0.8588**                  | 0.9169                     | 0.7355                    | 0.6895                     | 0.6112                    |
| 50.0K                | 4.0K / 4.0K                 | ✅                                     | 0.7918                     | 0.7706                      | 0.8042                     | 0.7481                    | 0.6644                     | 0.5412                    |
| 50.0K                | 10.0K / 10.0K               | ❌                                     | 0.8200                     | 0.8245                      | 0.8410                     | 0.7522                    | 0.6705                     | 0.5206                    |
| 50.0K                | 10.0K / 10.0K               | ✅                                     | 0.7652                     | 0.7717                      | 0.7665                     | **0.7789**                | 0.6860                     | **0.6322**                | 
| 100.0K               | 15.0K / 15.0K               | ✅                                     | 0.8155                     | 0.8175                      | 0.8142                     | 0.7213                    | 0.7106                     | 0.5795                    | 
| 300.0K               | 30.0K / 30.0K               | ❌                                     | **0.8579**                 | 0.8516                      | 0.8777                     | 0.7527                    | 0.6895                     | 0.6285                    |
| 500.0K               | 75.0K / 75.0K               | ✅                                     | 0.8561                     | 0.8365                      | 0.8622                     | 0.7713                    | **0.7244**                 | 0.5557                    |

