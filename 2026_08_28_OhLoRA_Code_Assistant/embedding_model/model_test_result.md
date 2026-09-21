
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

* 모델 교체 전/후: 기존 모델 `Giga-Embeddings-instruct-480M-0826` 대신 경량 모델인 `LateOn-Code-pretrain` 로 변경했을 때 전/후 성능 비교

| Oh-LoRA v7 규칙                           | 최종 적용 모델               | MAE<br>(모델 교체 후) | MSE<br>(모델 교체 후) | 실험 로그                                                                                                                                                  |
|-----------------------------------------|------------------------|------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 불필요한 print, logging 등이 없어야 함            | `gte-modernbert-base`  | 0.0073           | 0.0002           | [학습 로그](train_log/01_unnecessary_prints.csv), [테스트 로그](train_log/test_01_unnecessary_prints.csv)                                                       |
| 유사한 변수명은 하나로 통일 시키는 것이 좋음               | `LateOn-Code-pretrain` | 0.1115           | 0.0231           | [학습 로그](train_log/01_similar_variables.csv), [테스트 로그](train_log/test_01_similar_variables.csv)                                                         |
| 변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것)        | `LateOn-Code-pretrain` | 0.0509           | 0.0082           | [학습 로그](train_log/01_names.csv), [테스트 로그](train_log/test_01_names.csv)                                                                                 |
| 함수명과 반환값이 서로 잘 match 되어야 함              | `LateOn-Code-pretrain` | 0.1076           | 0.0182           | [학습 로그](train_log/01_return_matched_with_func_name.csv), [테스트 로그](train_log/test_01_return_matched_with_func_name.csv)                                 |
| 함수의 단일 책임 원칙 준수 여부 (docstring 으로 판단)    | `F2LLM-v2-330M`        | 0.0052           | 7.6e-5           | [학습 로그](train_log/01_func_docstring_single_responsibility.csv), [테스트 로그](train_log/test_01_func_docstring_single_responsibility.csv)                   |
| 함수 docstring과 함수명이 서로 일치하는지 판단          | `F2LLM-v2-330M`        | 0.1527           | 0.0387           | [학습 로그](train_log/01_func_docstring_docstring_and_name.csv), [테스트 로그](train_log/test_01_func_docstring_docstring_and_name.csv)                         |
| 함수의 인자가 하나로 묶을 수 있는 경우 처리 필요            | `LateOn-Code-pretrain` | 0.1057           | 0.0265           | [학습 로그](train_log/04_func_args_bindable.csv), [테스트 로그](train_log/test_04_func_args_bindable.csv)                                                       |
| 함수의 인자가 유동적인 경우 처리 필요                   | `LateOn-Code-pretrain` | 0.0652           | 0.0111           | [학습 로그](train_log/04_func_args_dynamic.csv), [테스트 로그](train_log/test_04_func_args_dynamic.csv)                                                         |
| 상태 값으로 판단되는 값을 조건으로 하는지 여부              | `LateOn-Code-pretrain` | 0.0331           | 0.0041           | [학습 로그](train_log/06_refactor_into_class_case_2_state_vars_if_else.csv), [테스트 로그](train_log/test_06_refactor_into_class_case_2_state_vars_if_else.csv) |
| 한 모듈 (*.py 파일) 내에서, 유사한 이름의 함수끼리 거리 검사  | `LateOn-Code-pretrain` | 0.1605           | 0.0396           | [학습 로그](train_log/06_similar_function_names.csv), [테스트 로그](train_log/test_06_similar_function_names.csv)                                               |
| 숫자 값을 constant 처럼 사용 시, 해당 값 상수화 적절성 판단 | `LateOn-Code-pretrain` | 0.1297           | 0.0348           | [학습 로그](train_log/02_numeric_values_maybe_const.csv), [테스트 로그](train_log/test_02_numeric_values_maybe_const.csv)                                       |
| 동일한 숫자 값 2회 이상 사용 시, 상수로 통합 적절성 판단      | `LateOn-Code-pretrain` | 0.1186           | 0.0226           | [학습 로그](train_log/02_numeric_values_twice.csv), [테스트 로그](train_log/test_02_numeric_values_twice.csv)                                                   |

## 2. 각 규칙 별 예측, 실제 점수 분포

### 2-1. 불필요한 print, logging 등이 없어야 함

![image](../../images/260828_6.PNG)

### 2-2. 유사한 변수명은 하나로 통일 시키는 것이 좋음

| 기존 모델<br>(`Giga-Embeddings-instruct-480M-0826`) | 변경 후 경량 모델<br>(`LateOn-Code-pretrain`) |
|-------------------------------------------------|----------------------------------------|
| ![image](../../images/260828_5.PNG)             | ![image](../../images/260828_11.PNG)   |

### 2-3. 변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것)

| 기존 모델<br>(`Giga-Embeddings-instruct-480M-0826`) | 변경 후 경량 모델<br>(`LateOn-Code-pretrain`) |
|-------------------------------------------------|----------------------------------------|
| ![image](../../images/260828_3.PNG)             | ![image](../../images/260828_12.PNG)   |

### 2-4. 함수명과 반환값이 서로 잘 match 되어야 함

| 기존 모델<br>(`Giga-Embeddings-instruct-480M-0826`) | 변경 후 경량 모델<br>(`LateOn-Code-pretrain`) |
|-------------------------------------------------|----------------------------------------|
| ![image](../../images/260828_4.PNG)             | ![image](../../images/260828_13.PNG)   |

### 2-5. 함수의 단일 책임 원칙 준수 여부 (docstring 으로 판단)

![image](../../images/260828_2.PNG)

### 2-6. 함수 docstring과 함수명이 서로 일치하는지 판단

![image](../../images/260828_1.PNG)

### 2-7. 함수의 인자가 하나로 묶을 수 있는 경우 처리 필요

| 기존 모델<br>(`Giga-Embeddings-instruct-480M-0826`) | 변경 후 경량 모델<br>(`LateOn-Code-pretrain`) |
|-------------------------------------------------|----------------------------------------|
| ![image](../../images/260828_7.PNG)             | ![image](../../images/260828_14.PNG)   |

### 2-8. 함수의 인자가 유동적인 경우 처리 필요

| 기존 모델<br>(`Giga-Embeddings-instruct-480M-0826`) | 변경 후 경량 모델<br>(`LateOn-Code-pretrain`) |
|-------------------------------------------------|----------------------------------------|
| ![image](../../images/260828_8.PNG)             | ![image](../../images/260828_15.PNG)   |

### 2-9. 상태 값으로 판단되는 값을 조건으로 하는지 여부

| 기존 모델<br>(`Giga-Embeddings-instruct-480M-0826`) | 변경 후 경량 모델<br>(`LateOn-Code-pretrain`) |
|-------------------------------------------------|----------------------------------------|
| ![image](../../images/260828_9.PNG)             | ![image](../../images/260828_16.PNG)   |

### 2-10. 한 모듈 (*.py 파일) 내에서, 유사한 이름의 함수끼리 거리 검사

| 기존 모델<br>(`Giga-Embeddings-instruct-480M-0826`) | 변경 후 경량 모델<br>(`LateOn-Code-pretrain`) |
|-------------------------------------------------|----------------------------------------|
| ![image](../../images/260828_10.PNG)            | ![image](../../images/260828_17.PNG)   |

### 2-11. 숫자 값을 constant 처럼 사용 시, 해당 값 상수화 적절성 판단

![image](../../images/260828_18.PNG)

### 2-12. 동일한 숫자 값 2회 이상 사용 시, 상수로 통합 적절성 판단

![image](../../images/260828_19.PNG)
