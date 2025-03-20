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

TBU

## 3. Odd-Radio Preference Optimizaiton (ORPO)

TBU