
## 목차

* [1. 요약](#1-요약)
* [2. dropout interval = 0.05, lr multiple = 0.8 & 1.25](#2-dropout-interval--005-lr-multiple--08--125)
* [3. dropout interval = 0.02, lr multiple = 0.9 & 1.111](#3-dropout-interval--002-lr-multiple--09--1111)

## 1. 요약

| Option                                                     | Pred. Macro F1 Score | True Macro F1 Score | HP find time |
|------------------------------------------------------------|----------------------|---------------------|--------------|
| dropout interval = **0.05**, lr multiple = **0.8 & 1.25**  | **0.6315**           | **0.6146**          | **2.7058 s** |
| dropout interval = **0.02**, lr multiple = **0.9 & 1.111** | 0.6135               | 0.5928              | 5.7112 s     |

* 결론
  * dropout interval 과 lr (learning rate) 에 매 탐색 step 마다 곱하는 multiple 값을 더 촘촘하게 하면, **Macro F1 Score (성능지표) 및 하이퍼파라미터 탐색 시간** 관점에서 모두 **불리하다**

## 2. dropout interval = 0.05, lr multiple = 0.8 & 1.25

* from [test_result.csv](test_result.csv)

**Average Result**

| Dataset       | Pred. Macro F1 Score | True Macro F1 Score | HP find time |
|---------------|----------------------|---------------------|--------------|
| **전체 데이터셋**   | **0.6315**           | **0.6146**          | **2.7058 s** |
| CIFAR-10      | 0.3669               | 0.3861              | 3.2655 s     |
| Fashion MNIST | 0.7438               | 0.6917              | 2.6111 s     |
| MNIST         | 0.7838               | 0.7659              | 2.2409 s     |

## 3. dropout interval = 0.02, lr multiple = 0.9 & 1.111

* from [test_result_narrow.csv](test_result_narrow.csv)

**Average Result**

| Dataset       | Pred. Macro F1 Score | True Macro F1 Score | HP find time |
|---------------|----------------------|---------------------|--------------|
| **전체 데이터셋**   | **0.6135**           | **0.5928**          | **5.7112 s** |
| CIFAR-10      | 0.3923               | 0.4031              | 6.6391 s     |
| Fashion MNIST | 0.7181               | 0.6697              | 5.7777 s     |
| MNIST         | 0.7303               | 0.7056              | 4.7168 s     |
