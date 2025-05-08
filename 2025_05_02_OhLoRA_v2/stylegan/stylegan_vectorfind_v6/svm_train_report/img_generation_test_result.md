## Image Generation Test Result

* with **[sklearnex](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)** for all cases
* always used ```LinearSVC(...)``` instead of ```SVC(kernel='linear', ...)```
* performance (accuracy of ```eyes```, ```mouth``` and ```pose```) means **SVM accuracy**
* for **mean corr-coef**,
  * each corr-coef means the corr-coef of **Intended Property Scores vs. Actual CNN-Predicted Property Scores** for 50 generated images

| n<br>(total samples) | k<br>(top / bottom samples) | grouping<br>(8 groups) | ```eyes``` acc.<br>(0 ~ 1) | ```mouth``` acc.<br>(0 ~ 1) | ```pose``` acc.<br>(0 ~ 1) | ```eyes``` mean corr-coef | ```mouth``` mean corr-coef | ```pose``` mean corr-coef |
|----------------------|-----------------------------|------------------------|----------------------------|-----------------------------|----------------------------|---------------------------|----------------------------|---------------------------|
| 4.0K                 | 800 / 800                   | ❌                      | 0.7000                     | 0.7094                      | 0.7656                     | 0.7348                    | 0.6267                     | 0.5610                    |
| 4.0K                 | 800 / 800                   | ✅                      | 0.6135                     | 0.6472                      | 0.6902                     | 0.6744                    | 0.5504                     | 0.5963                    |
| 50.0K                | 800 / 800                   | ❌                      | 0.8156                     | 0.8187                      | **0.9219**                 | 0.7081                    | **0.6971**                 | 0.5206                    |
| 50.0K                | 4.0K / 4.0K                 | ❌                      | 0.8525                     | **0.8588**                  | 0.9169                     | 0.7355                    | 0.6895                     | 0.6112                    |
| 50.0K                | 4.0K / 4.0K                 | ✅                      | 0.7918                     | 0.7706                      | 0.8042                     | 0.7481                    | 0.6644                     | 0.5412                    |
| 50.0K                | 10.0K / 10.0K               | ❌                      | 0.8200                     | 0.8245                      | 0.8410                     | 0.7522                    | 0.6705                     | 0.5206                    |
| 50.0K                | 10.0K / 10.0K               | ✅                      | 0.7652                     | 0.7717                      | 0.7665                     | **0.7789**                | 0.6860                     | **0.6322**                | 
| 300.0K               | 30.0K / 30.0K               | ❌                      | **0.8579**                 | 0.8516                      | 0.8777                     | 0.7527                    | 0.6895                     | 0.6285                    |