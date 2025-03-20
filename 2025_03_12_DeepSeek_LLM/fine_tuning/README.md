## 목차

* [1. LLM Fine-Tuning](#1-llm-fine-tuning)
* [2. Supervised Fine-Tuning (SFT)](#2-supervised-fine-tuning-sft)
* [3. Odd-Radio Preference Optimizaiton (ORPO)](#3-odd-radio-preference-optimizaiton-orpo)

## 1. LLM Fine-Tuning

**1. LLM Fine-Tuning 방법 요약**

* LLM
  * **deepseek-coder-1.3b-instruct**
* Supervised Fine-Tuning (SFT)
  * 지도학습을 통해 모델이 diagram format 에 맞는 text 를 생성하도록 유도 
* Odd-Radio Preference Optimizaiton (ORPO)
  * 유저가 선호하는 스타일의 diagram 이 생성되는 text 를 생성하도록 유도 

![image](../../images/250312_4.PNG)

**2. Fine-Tuning 상세**

| 방법                                       | 데이터셋                                                                                                                                              | 데이터셋 파일                                                        |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Supervised Fine-Tuning (SFT)             | **총 700 개 규모** 데이터셋<br>- 딥러닝 모델 구조 관련 280 개<br>- 기타 420 개                                                                                         | [sft_dataset_llm.csv](../create_dataset/sft_dataset_llm.csv)   |
| Odd-Radio Preference Optimizaiton (ORPO) | **총 800 개 규모** 데이터셋<br>- 딥러닝 모델 구조 관련 320 개 (80 개는 SFT 이전 생성, 240 개는 SFT 된 LLM 에 의해 생성)<br>- 기타 480개 (120 개는 SFT 이전 생성, 360 개는 SFT 된 LLM 에 의해 생성) | [orpo_dataset_llm.csv](../create_dataset/orpo_dataset_llm.csv) |

## 2. Supervised Fine-Tuning (SFT)

**1. 데이터셋**

* ```create_dataset/sft_dataset_llm.csv```
* 총 700 개 규모 데이터셋
  * 딥러닝 모델 구조 관련 280 개
  * 기타 420 개

**2. Fine-Tuning 코드**

* 학습 코드
  * ```fine_tuning/sft_fine_tuning.py```
* SFT 된 모델을 이용한 ORPO 용 데이터 생성 코드
  * ```fine_tuning/orpo_create_dataset.py```
  * 해당 코드 실행 시, ```create_dataset/orpo_dataset_llm.csv``` 에 ORPO 용 추가 데이터를 생성하여 추가

**3. Fine-Tuning 결과**

TBU

## 3. Odd-Radio Preference Optimizaiton (ORPO)

**1. 데이터셋**

* ```create_dataset/orpo_dataset_llm.csv```
  * **SFT 된 모델로 ORPO 용 추가 데이터를 생성한 후** 의 csv 파일이어야 함
* 다음과 같이 총 800 개 규모 데이터셋을 바탕으로 ORPO 학습 데이터셋 생성
  * 딥러닝 모델 구조 관련 320 개
    * 80 개는 SFT 이전 생성
    * 240 개는 SFT 된 LLM 에 의해 생성
  * 기타 480 개
    * 120 개는 SFT 이전 생성
    * 360 개는 SFT 된 LLM 에 의해 생성

**2. Fine-Tuning 코드**

* ```fine_tuning/orpo_fine_tuning.py```

**3. Fine-Tuning 결과**

TBU