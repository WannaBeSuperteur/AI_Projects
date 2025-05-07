## 목차

* [1. Key Points](#1-key-points)
  * [1-1. Time Complexity (추정)](#1-1-time-complexity-추정) 
  * [1-2. Speedup of ```sklearnex``` Library](#1-2-speedup-of-sklearnex-library)
* [2. Performance & Training Time Record](#2-performance--training-time-record)

## 1. Key Points

* 데이터 샘플 개수가 많을수록 각 Property Score 에 대한 성능 (SVM 정확도) 이 높아지는 추세이다.

### 1-1. Time Complexity (추정)

| 알고리즘                                                 | Time Complexity<br>(실험 결과 기반 추정) |
|------------------------------------------------------|----------------------------------|
| **Sampling (Synthesize)** from latent vector z       | $O(n)$                           |
| **t-SNE**                                            | $O(n)$                           |
| **SVM** (Support Vector Machine) - **linear** kernel | $O(n^2)$ 또는 그 이상                 |

* SVM implementation

```python
svm_clf = svm.SVC(kernel='linear', random_state=2025+i)
```

### 1-2. Speedup of ```sklearnex``` Library

TBU

## 2. Performance & Training Time Record

* performance (accuracy of ```eyes```, ```mouth``` and ```pose```) means **SVM accuracy**
* performance recorded only when $\displaystyle \frac{k}{n}$ is always **0.2 (= 20.0%)**

| Options                  |                                 |                                                                                                                      | Time (sec)                |                                      |                                             | Performance                    |                                 |                                |
|--------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------|--------------------------------------|---------------------------------------------|--------------------------------|---------------------------------|--------------------------------|
| **n**<br>(total samples) | **k**<br>(top / bottom samples) | **with [sklearnex](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)** | **Sampling (Synthesize)** | **t-SNE**<br>(all 3 property scores) | **SVM training**<br>(all 3 property scores) | **```eyes``` acc.<br>(0 ~ 1)** | **```mouth``` acc.<br>(0 ~ 1)** | **```pose``` acc.<br>(0 ~ 1)** |
| 1.0K                     | 200 / 200                       | ❌                                                                                                                    | 37.1                      | 6.6                                  | 0.1                                         | 0.6750                         | 0.7375                          | 0.6250                         |
| 2.0K                     | 400 / 400                       | ❌                                                                                                                    | 84.2                      | 15.3                                 | 0.4                                         | 0.5938                         | 0.6875                          | 0.6625                         |
| 4.0K                     | 800 / 800                       | ❌                                                                                                                    | 156.6                     | 21.1                                 | 1.8                                         | 0.6750                         | 0.7156                          | 0.7344                         |
| 10.0K                    | 2.0K / 2.0K                     | ❌                                                                                                                    | 391.5                     | 53.6                                 | 152.2                                       | 0.7788                         | 0.8100                          | 0.7850                         |
| 25.0K                    | 5.0K / 5.0K                     | ❌                                                                                                                    | 944.5                     | 141.2                                | 514.6                                       | 0.8175                         | 0.8295                          | 0.8485                         |
| 25.0K                    | 5.0K / 5.0K                     | ✅                                                                                                                    | 1084.8                    | 204.0                                | 347.6                                       | 0.8150                         | 0.8110                          | 0.8385                         |
| 50.0K                    | 10.0K / 10.0K                   | ✅                                                                                                                    | 2110.0                    | 442.6                                | 538.0                                       | 0.8472                         | 0.8260                          | 0.8538                         |
| 50.0K                    | 10.0K / 10.0K                   | ✅ + LinearSVC                                                                                                        | 2142.2                    | 435.6                                | 63.2                                        | 0.8200                         | 0.8245                          | 0.8410                         |
| 300.0K                   | 30.0K / 30.0K                   | ✅ + LinearSVC                                                                                                        | 11788.2                   | 491.6                                | 229.6                                       |                                |                                 |                                |