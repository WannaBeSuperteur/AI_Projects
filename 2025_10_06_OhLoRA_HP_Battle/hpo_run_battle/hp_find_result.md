
## 목차

* [1. 요약](#1-요약)
* [2. dropout interval = 0.05, lr multiple = 0.8 & 1.25](#2-dropout-interval--005-lr-multiple--08--125)
* [3. dropout interval = 0.02, lr multiple = 0.9 & 1.111](#3-dropout-interval--002-lr-multiple--09--1111)
* [4. Optuna 와의 대결 결과](#4-optuna-와의-대결-결과)

## 1. 요약

| Option                                                     | Pred. Macro F1 Score | True Macro F1 Score | HP find time |
|------------------------------------------------------------|----------------------|---------------------|--------------|
| dropout interval = **0.05**, lr multiple = **0.8 & 1.25**  | **0.6315**           | **0.6146**          | **2.7058 s** |
| dropout interval = **0.02**, lr multiple = **0.9 & 1.111** | 0.6135               | 0.5928              | 5.7112 s     |

* 결론
  * dropout interval 과 lr (learning rate) 에 매 탐색 step 마다 곱하는 multiple 값을 더 촘촘하게 하면, **Macro F1 Score (성능지표) 및 하이퍼파라미터 탐색 시간** 관점에서 모두 **불리하다.**

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

## 4. Optuna 와의 대결 결과

**각 데이터셋 별 총 30회** Optuna와 대결 실시

* 결론
  * Oh-LoRA 는 **Optuna Trials = 4** 정도의 성능을 보임 

* 승률
  * 무승부의 경우 **Oh-LoRA 0.5승** 으로 처리

| Dataset       | Optuna Trials=2 | Optuna Trials=5 | Optuna Trials=15 |
|---------------|-----------------|-----------------|------------------|
| **전체**        | **68.3%**       | **42.8%**       | **34.4%**        |
| CIFAR-10      | 70.0%           | 58.3%           | 33.3%            |
| Fashion MNIST | 73.3%           | 36.7%           | 40.0%            |
| MNIST         | 61.7%           | 33.3%           | 30.0%            |

* 평균 True Macro F1 Score 차이
  * = **{Oh-LoRA 점수} - {Optuna 점수}**

| Dataset       | Optuna Trials=2 | Optuna Trials=5 | Optuna Trials=15 |
|---------------|-----------------|-----------------|------------------|
| **전체**        | **+0.0652**     | **-0.0201**     | **-0.0313**      |
| CIFAR-10      | +0.0661         | +0.0157         | -0.0489          |
| Fashion MNIST | +0.0556         | -0.0994         | -0.0075          |
| MNIST         | +0.0739         | +0.0235         | -0.0374          |

* 상세 정보
  * [Optuna Trials = 2](test_result_vs_optuna_2_trials.csv)
  * [Optuna Trials = 5](test_result_vs_optuna_5_trials.csv)
  * [Optuna Trials = 15](test_result_vs_optuna_15_trials.csv)
