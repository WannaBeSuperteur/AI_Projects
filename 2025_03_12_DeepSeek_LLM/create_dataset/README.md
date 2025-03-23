## 목차

* [1. ```sft_dataset.py``` 설명](#1-sftdatasetpy-설명)
* [2. ```orpo_dataset.py``` 설명](#2-orpodatasetpy-설명)
* [3. ```common.py``` 설명](#3-commonpy-설명)
* [4. ```convert_to_llm_dataset.py``` 설명](#4-converttollmdatasetpy-설명)
* [5. ```test_draw_diagram.py``` 설명](#5-testdrawdiagrampy-설명)

## 개요

![image](../../images/250312_3.PNG)

## 1. ```sft_dataset.py``` 설명

* [Supervised Fine-Tuning (SFT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md) 용 데이터셋 생성 코드
  * 생성된 데이터셋을 ```sft_dataset.csv``` 에 저장
* 총 700 개 규모 데이터셋
  * 딥러닝 모델 구조 관련 280 개
  * 기타 420 개

## 2. ```orpo_dataset.py``` 설명

* [ORPO (Odd-Ratio Preference Optimization)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_DPO_ORPO.md#3-orpo-odds-ratio-preference-optimization) 용 데이터셋 생성 코드
  * 생성된 데이터셋을 ```orpo_dataset.csv``` 에 저장
* 총 199 개 규모 데이터셋을 바탕으로 ORPO 학습 데이터셋 생성
  * 원래 LLM 이 목표로 하는 output 을 chosen 으로 간주
  * SFT 로 Fine-Tuning 된 모델을 이용하여 생성한 output 을 rejected 로 간주
  * 200개 데이터 중 1개는 SFT 로 Fine-Tuning 된 모델이 출력한 output 에 대한 점수가 만점이므로, rejected 로 간주 불가하여 데이터셋에서 제외

## 3. ```common.py``` 설명

* SFT, ORPO 에 공통으로 사용되는 형태의 데이터셋 생성의 경우, 관련 함수를 common.py 에 정의

## 4. ```convert_to_llm_dataset.py``` 설명

* csv 형태의 데이터셋 파일을 실제 LLM 이 학습 가능한 형태의 데이터셋으로 변환
* csv 형태의 데이터셋 파일
  * ```sft_dataset.csv```
  * ```orpo_dataset.csv```

## 5. ```test_draw_diagram.py``` 설명

* 해당 파일 실행 시, ```sft_dataset.csv``` 파일의 각 내용 (row) 을 읽어서 테스트 다이어그램 생성
* 테스트 다이어그램은 ```test_diagrams``` 디렉토리에 생성