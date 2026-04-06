# HPO model cutoff 테스트 결과

## 목차

* [1. 테스트 목적](#1-테스트-목적)
* [2. 테스트 결과](#2-테스트-결과)
* [3. 테스트 결과에 따른 결정](#3-테스트-결과에-따른-결정)

## 1. 테스트 목적

* Hyper-parameter 최적화 모델의 학습 데이터는 Tabular Dataset 이다.
* 이 데이터셋의 input feature 중 **target feature (Macro F1-score) 와의 상관계수의 절댓값이 일정 값 (= cutoff) 이상** 인 feature만 모델 학습에 사용한다.
* 이 **cutoff 값에 따른 HPO 모델의 성능 추이를 관측** 하고, 이를 통해 각 데이터셋 별 **최선의 cutoff 값을 찾는다.**

## 2. 테스트 결과

* MSE (Mean-Squared Error)

* MAE (Mean Absolute Error)

* feature 개수 (target feature 와의 corr-coef 가 해당 cutoff 값 이상인)

## 3. 테스트 결과에 따른 결정

* TBU