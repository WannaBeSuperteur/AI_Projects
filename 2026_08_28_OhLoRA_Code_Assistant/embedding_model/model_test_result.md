
## 목차

* [1. 개요](#1-개요)
* [2. 각 규칙 별 예측, 실제 점수 분포](#2-각-규칙-별-예측-실제-점수-분포)
  * [2-1. 불필요한 print, logging 등이 없어야 함](#2-1-불필요한-print-logging-등이-없어야-함) 
  * [2-2. 유사한 변수명은 하나로 통일 시키는 것이 좋음](#2-2-유사한-변수명은-하나로-통일-시키는-것이-좋음)
  * [2-3. 변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것)](#2-3-변수명-함수명은-의미가-있어야-함--알기-쉽게-할것)
  * [2-4. 함수명과 반환값이 서로 잘 match 되어야 함](#2-4-함수명과-반환값이-서로-잘-match-되어야-함)
  * [2-5. 함수의 단일 책임 원칙 준수 여부 (docstring 으로 판단)](#2-5-함수의-단일-책임-원칙-준수-여부-docstring-으로-판단)
  * [2-6. 함수 docstring과 함수명이 서로 일치하는지 판단](#2-6-함수-docstring과-함수명이-서로-일치하는지-판단)
  * [2-7. 함수의 인자가 하나로 묶을 수 있는 경우 처리 필요](#2-7-함수의-인자가-하나로-묶을-수-있는-경우-처리-필요)
  * [2-8. 함수의 인자가 유동적인 경우 처리 필요](#2-8-함수의-인자가-유동적인-경우-처리-필요)
  * [2-9. 상태 값으로 판단되는 값을 조건으로 하는지 여부](#2-9-상태-값으로-판단되는-값을-조건으로-하는지-여부)
  * [2-10. 한 모듈 (*.py 파일) 내에서, 유사한 이름의 함수끼리 거리 검사](#2-10-한-모듈-py-파일-내에서-유사한-이름의-함수끼리-거리-검사)
  * [2-11. 숫자 값을 constant 처럼 사용 시, 해당 값 상수화 적절성 판단](#2-11-숫자-값을-constant-처럼-사용-시-해당-값-상수화-적절성-판단)
  * [2-12. 동일한 숫자 값 2회 이상 사용 시, 상수로 통합 적절성 판단](#2-12-동일한-숫자-값-2회-이상-사용-시-상수로-통합-적절성-판단)

## 1. 개요

* 모델 교체 후: 기존 모델 `Giga-Embeddings-instruct-480M-0826` 대신 경량 모델인 `LateOn-Code-pretrain` 로 변경했을 때 전/후 성능 비교
* more epochs: 다음과 같이 epoch 횟수 상향 조정
* bugfix: 기존에 **last epoch model** 기준으로 test 했던 것을 **best epoch model** 기준으로 test 하도록 수정

| 구분                                | more epochs 미 적용 | more epochs 적용 |
|-----------------------------------|------------------|----------------|
| 확률 예측 모델 max epochs               | 20               | 50             |
| 확률 예측 모델 early stopping patience  | 5                | 10             |
| 유사도 측정 모델 max epochs              | 12               | 20             |
| 유사도 측정 모델 early stopping patience | 3                | 3              |

* 각 규칙 별 모델, 실험 결과 (성능), 상세 실험 로그
  * 최종 사용 버전: 모든 Oh-LoRA v7 규칙에 대해 **모델 교체 & more epochs & bugfix** 버전 

| Oh-LoRA v7 규칙                           | 최종 적용 모델               | MAE<br>(모델 교체 후 / 모델 교체 & more epochs & bugfix) | MSE<br>(모델 교체 후 / 모델 교체 & more epochs & bugfix) | 실험 로그                                                                                                                                                  |
|-----------------------------------------|------------------------|-------------------------------------------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 불필요한 print, logging 등이 없어야 함            | `gte-modernbert-base`  | 0.0073 / **0.0069**                             | 0.0002 / **0.0003**                             | [학습 로그](train_log/01_unnecessary_prints.csv), [테스트 로그](train_log/test_01_unnecessary_prints.csv)                                                       |
| 유사한 변수명은 하나로 통일 시키는 것이 좋음               | `LateOn-Code-pretrain` | 0.1115 / **0.1042**                             | 0.0231 / **0.0189**                             | [학습 로그](train_log/01_similar_variables.csv), [테스트 로그](train_log/test_01_similar_variables.csv)                                                         |
| 변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것)        | `LateOn-Code-pretrain` | 0.0509 / **0.0573**                             | 0.0082 / **0.0078**                             | [학습 로그](train_log/01_names.csv), [테스트 로그](train_log/test_01_names.csv)                                                                                 |
| 함수명과 반환값이 서로 잘 match 되어야 함              | `LateOn-Code-pretrain` | 0.1076 / **0.1011**                             | 0.0182 / **0.0174**                             | [학습 로그](train_log/01_return_matched_with_func_name.csv), [테스트 로그](train_log/test_01_return_matched_with_func_name.csv)                                 |
| 함수의 단일 책임 원칙 준수 여부 (docstring 으로 판단)    | `F2LLM-v2-330M`        | 0.0052 / **0.0034**                             | 7.6e-5 / **4.3e-5**                             | [학습 로그](train_log/01_func_docstring_single_responsibility.csv), [테스트 로그](train_log/test_01_func_docstring_single_responsibility.csv)                   |
| 함수 docstring과 함수명이 서로 일치하는지 판단          | `F2LLM-v2-330M`        | 0.1527 / **0.1354**                             | 0.0387 / **0.0289**                             | [학습 로그](train_log/01_func_docstring_docstring_and_name.csv), [테스트 로그](train_log/test_01_func_docstring_docstring_and_name.csv)                         |
| 함수의 인자가 하나로 묶을 수 있는 경우 처리 필요            | `LateOn-Code-pretrain` | 0.1057 / **0.0956**                             | 0.0265 / **0.0175**                             | [학습 로그](train_log/04_func_args_bindable.csv), [테스트 로그](train_log/test_04_func_args_bindable.csv)                                                       |
| 함수의 인자가 유동적인 경우 처리 필요                   | `LateOn-Code-pretrain` | 0.0652 / **0.0698**                             | 0.0111 / **0.0106**                             | [학습 로그](train_log/04_func_args_dynamic.csv), [테스트 로그](train_log/test_04_func_args_dynamic.csv)                                                         |
| 상태 값으로 판단되는 값을 조건으로 하는지 여부              | `LateOn-Code-pretrain` | 0.0331 / **0.0410**                             | 0.0041 / **0.0072**                             | [학습 로그](train_log/06_refactor_into_class_case_2_state_vars_if_else.csv), [테스트 로그](train_log/test_06_refactor_into_class_case_2_state_vars_if_else.csv) |
| 한 모듈 (*.py 파일) 내에서, 유사한 이름의 함수끼리 거리 검사  | `LateOn-Code-pretrain` | 0.1605 / **0.1602**                             | 0.0396 / **0.0452**                             | [학습 로그](train_log/06_similar_function_names.csv), [테스트 로그](train_log/test_06_similar_function_names.csv)                                               |
| 숫자 값을 constant 처럼 사용 시, 해당 값 상수화 적절성 판단 | `LateOn-Code-pretrain` | 0.1297 / **0.1131**                             | 0.0348 / **0.0197**                             | [학습 로그](train_log/02_numeric_values_maybe_const.csv), [테스트 로그](train_log/test_02_numeric_values_maybe_const.csv)                                       |
| 동일한 숫자 값 2회 이상 사용 시, 상수로 통합 적절성 판단      | `LateOn-Code-pretrain` | 0.1186 / **0.1730**                             | 0.0226 / **0.0433**                             | [학습 로그](train_log/02_numeric_values_twice.csv), [테스트 로그](train_log/test_02_numeric_values_twice.csv)                                                   |

## 2. 각 규칙 별 예측, 실제 점수 분포

### 2-1. 불필요한 print, logging 등이 없어야 함

| 모델                                  | 실험 결과                                |
|-------------------------------------|--------------------------------------|
| `gte-modernbert-base`               | ![image](../../images/260828_6.PNG)  |
| `gte-modernbert-base` (more epochs) | ![image](../../images/260828_20.PNG) |

### 2-2. 유사한 변수명은 하나로 통일 시키는 것이 좋음

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `Giga-Embeddings-instruct-480M-0826` | ![image](../../images/260828_5.PNG)  |
| `LateOn-Code-pretrain`               | ![image](../../images/260828_11.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_21.PNG) |

### 2-3. 변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것)

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `Giga-Embeddings-instruct-480M-0826` | ![image](../../images/260828_3.PNG)  |
| `LateOn-Code-pretrain`               | ![image](../../images/260828_12.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_22.PNG) |

### 2-4. 함수명과 반환값이 서로 잘 match 되어야 함

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `Giga-Embeddings-instruct-480M-0826` | ![image](../../images/260828_4.PNG)  |
| `LateOn-Code-pretrain`               | ![image](../../images/260828_13.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_23.PNG) |

### 2-5. 함수의 단일 책임 원칙 준수 여부 (docstring 으로 판단)

| 모델                            | 실험 결과                                |
|-------------------------------|--------------------------------------|
| `F2LLM-v2-330M`               | ![image](../../images/260828_2.PNG)  |
| `F2LLM-v2-330M` (more epochs) | ![image](../../images/260828_24.PNG) |

### 2-6. 함수 docstring과 함수명이 서로 일치하는지 판단

| 모델                            | 실험 결과                                |
|-------------------------------|--------------------------------------|
| `F2LLM-v2-330M`               | ![image](../../images/260828_1.PNG)  |
| `F2LLM-v2-330M` (more epochs) | ![image](../../images/260828_25.PNG) |

### 2-7. 함수의 인자가 하나로 묶을 수 있는 경우 처리 필요

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `Giga-Embeddings-instruct-480M-0826` | ![image](../../images/260828_7.PNG)  |
| `LateOn-Code-pretrain`               | ![image](../../images/260828_14.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_26.PNG) |

### 2-8. 함수의 인자가 유동적인 경우 처리 필요

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `Giga-Embeddings-instruct-480M-0826` | ![image](../../images/260828_8.PNG)  |
| `LateOn-Code-pretrain`               | ![image](../../images/260828_15.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_27.PNG) |

### 2-9. 상태 값으로 판단되는 값을 조건으로 하는지 여부

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `Giga-Embeddings-instruct-480M-0826` | ![image](../../images/260828_9.PNG)  |
| `LateOn-Code-pretrain`               | ![image](../../images/260828_16.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_28.PNG) |

### 2-10. 한 모듈 (*.py 파일) 내에서, 유사한 이름의 함수끼리 거리 검사

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `Giga-Embeddings-instruct-480M-0826` | ![image](../../images/260828_10.PNG) |
| `LateOn-Code-pretrain`               | ![image](../../images/260828_17.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_29.PNG) |

### 2-11. 숫자 값을 constant 처럼 사용 시, 해당 값 상수화 적절성 판단

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `LateOn-Code-pretrain`               | ![image](../../images/260828_18.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_30.PNG) |

### 2-12. 동일한 숫자 값 2회 이상 사용 시, 상수로 통합 적절성 판단

| 모델                                   | 실험 결과                                |
|--------------------------------------|--------------------------------------|
| `LateOn-Code-pretrain`               | ![image](../../images/260828_19.PNG) |
| `LateOn-Code-pretrain` (more epochs) | ![image](../../images/260828_31.PNG) |
