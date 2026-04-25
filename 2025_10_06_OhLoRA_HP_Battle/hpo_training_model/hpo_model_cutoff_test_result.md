# HPO model cutoff 테스트 결과

## 목차

* [1. 테스트 목적](#1-테스트-목적)
* [2. Option 설명](#2-option-설명)
* [3. 테스트 결과](#3-테스트-결과)
  * [3-1. Option 1 테스트 결과](#3-1-option-1-테스트-결과)
  * [3-2. Option 2 테스트 결과](#3-2-option-2-테스트-결과)
  * [3-3. Option 3 테스트 결과](#3-3-option-3-테스트-결과)
* [4. 테스트 결과에 따른 결정](#4-테스트-결과에-따른-결정)
  * [4-1. Option 1 테스트 결과](#4-1-option-1-테스트-결과)
  * [4-2. Option 2 테스트 결과](#4-2-option-2-테스트-결과)
  * [4-3. Option 3 테스트 결과](#4-3-option-3-테스트-결과)

## 1. 테스트 목적

* Hyper-parameter 최적화 모델의 학습 데이터는 Tabular Dataset 이다.
* 이 데이터셋의 input feature 중 **target feature (Macro F1-score) 와의 상관계수의 절댓값이 일정 값 (= cutoff) 이상** 인 feature만 모델 학습에 사용한다.
* 이 **cutoff 값에 따른 HPO 모델의 성능 추이를 관측** 하고, 이를 통해 각 데이터셋 별 **최선의 cutoff 값을 찾는다.**

## 2. Option 설명

## 3. 테스트 결과

* 테스트 결과 요약

### 3-1. Option 1 테스트 결과

각 데이터셋 별로 **직사각형으로 표시한 cutoff threshold 구간 (가로축)** 에서 Error 가 가장 작음

* MSE (Mean-Squared Error)

![image](../../images/251006_7.PNG)

* MAE (Mean Absolute Error)

![image](../../images/251006_8.png)

* feature 개수 (target feature 와의 corr-coef 가 해당 cutoff 값 이상인)

![image](../../images/251006_9.png)

### 3-2. Option 2 테스트 결과

### 3-3. Option 3 테스트 결과

## 4. 테스트 결과에 따른 결정

### 4-1. Option 1 테스트 결과

* 각 데이터셋 별로 다음과 같은 threshold cutoff 를 이용하여 모델 학습

| cifar_10 | fashion_mnist | mnist |
|----------|---------------|-------|
| 0.20     | 0.175         | 0.35  |

### 4-2. Option 2 테스트 결과

* Option 3으로 테스트 재 실시

### 4-3. Option 3 테스트 결과
