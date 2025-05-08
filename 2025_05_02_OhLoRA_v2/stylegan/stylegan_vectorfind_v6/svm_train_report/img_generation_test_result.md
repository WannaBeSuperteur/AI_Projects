## Image Generation Test Result

* with **[sklearnex](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)** for all cases
* always used ```LinearSVC(...)``` instead of ```SVC(kernel='linear', ...)```
* performance (accuracy of ```eyes```, ```mouth``` and ```pose```) means **SVM accuracy**
* for **mean corr-coef**,
  * each corr-coef means the corr-coef of **Intended Property Scores vs. Actual CNN-Predicted Property Scores** for 50 generated images

| n<br>(total samples) | k<br>(top / bottom samples) | ```eyes``` acc.<br>(0 ~ 1) | ```mouth``` acc.<br>(0 ~ 1) | ```pose``` acc.<br>(0 ~ 1) | ```eyes``` mean corr-coef | ```mouth``` mean corr-coef | ```pose``` mean corr-coef |
|----------------------|-----------------------------|----------------------------|-----------------------------|----------------------------|---------------------------|----------------------------|---------------------------|
| 4.0K                 | 800 / 800                   | 0.7000                     | 0.7094                      | 0.7656                     | 0.7348                    | 0.6267                     | 0.5610                    |
| 50.0K                | 800 / 800                   | 0.8156                     | 0.8187                      | 0.9219                     | 0.7081                    | 0.6971                     | 0.5206                    |
| 50.0K                | 4.0K / 4.0K                 | 0.8525                     | 0.8588                      | 0.9169                     | 0.7355                    | 0.6895                     | 0.6112                    |
| 50.0K                | 10.0K / 10.0K               | 0.8200                     | 0.8245                      | 0.8410                     | 0.7522                    | 0.6705                     | 0.5206                    |
| 300.0K               | 30.0K / 30.0K               | 0.8579                     | 0.8516                      | 0.8777                     | 0.7527                    | 0.6895                     | 0.6285                    |

## Grouping

* 이미지를 random latent code (z) 로부터 생성할 때,
  * 해당 이미지의 **머리 색, 머리 길이, 배경 색 평균** 의 핵심 속성 값을 Property Score CNN 으로 예측
  * 해당 핵심 속성 값을 기준으로 이미지를 각 그룹으로 분류
* 각 그룹별로,
  * t-SNE 실시 및 SVM 학습
  * **눈을 뜬 정도, 입을 벌린 정도, 고개 돌림** 각각에 대한 n vector 계산
* inference 시,
  * 생성된 이미지가 어느 그룹에 속하는지 Property Score CNN 으로 판단
  * 해당 그룹에 맞는 **눈을 뜬 정도, 입을 벌린 정도, 고개 돌림** n vector 를 이용하여 핵심 속성 값이 변동된 이미지 생성