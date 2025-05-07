## Image Generation Test Result

* with **[sklearnex](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)** for all cases
* always used ```LinearSVC(...)``` instead of ```SVC(kernel='linear', ...)```
* performance (accuracy of ```eyes```, ```mouth``` and ```pose```) means **SVM accuracy**
* for **mean corr-coef**,
  * each corr-coef means the corr-coef of **Intended Property Scores vs. Actual CNN-Predicted Property Scores** for 50 generated images

| n<br>(total samples) | k<br>(top / bottom samples) | ```eyes``` acc.<br>(0 ~ 1) | ```mouth``` acc.<br>(0 ~ 1) | ```pose``` acc.<br>(0 ~ 1) | ```eyes``` mean corr-coef | ```mouth``` mean corr-coef | ```pose``` mean corr-coef |
|----------------------|-----------------------------|----------------------------|-----------------------------|----------------------------|---------------------------|----------------------------|---------------------------|
| 4.0K                 | 800 / 800                   | 0.7000                     | 0.7094                      | 0.7656                     | 0.7348                    | 0.6267                     | 0.5610                    |
| 50.0K                | 10.0K / 10.0K               | 0.8200                     | 0.8245                      | 0.8410                     | 0.7522                    | 0.6705                     | 0.5206                    |