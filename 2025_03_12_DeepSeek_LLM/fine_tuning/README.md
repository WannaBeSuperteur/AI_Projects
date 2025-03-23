## 목차

* [1. LLM Fine-Tuning](#1-llm-fine-tuning)
* [2. Supervised Fine-Tuning (SFT)](#2-supervised-fine-tuning-sft)
* [3. Odd-Radio Preference Optimizaiton (ORPO) **(실패, 미 적용)**](#3-odd-radio-preference-optimizaiton-orpo)
* [4. 코드 실행 순서](#4-코드-실행-순서)

## 1. LLM Fine-Tuning

**1. LLM Fine-Tuning 방법 요약**

* LLM
  * **deepseek-coder-1.3b-instruct**
  * [14개 후보 LLM 대상 테스트 결과, 해당 LLM 이 성능 및 속도 측면에서 가장 우수](../test_llm/README.md#3-테스트-진행-및-결과)
* Fine-Tuning 방법
  * [Supervised Fine-Tuning (SFT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)
    * 지도학습을 통해 모델이 diagram format 에 맞는 text 를 생성하도록 유도
    * [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md#2-lora-low-rank-adaptation) 를 선택
  * [Odd-Radio Preference Optimizaiton (ORPO)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_DPO_ORPO.md#3-orpo-odds-ratio-preference-optimization)
    * 유저가 선호하는 스타일의 diagram 이 생성되는 text 를 생성하도록 유도 

![image](../../images/250312_4.PNG)

**2. Fine-Tuning 상세**

* SFT, ORPO 각각 데이터셋 중 **80% 를 train, 20% 를 validation** 에 사용 (test dataset 따로 없음)

| 방법                                       | 데이터셋                                                      | 데이터셋 파일                                                        |
|------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------|
| Supervised Fine-Tuning (SFT)             | **총 700 개 규모** 데이터셋<br>- 딥러닝 모델 구조 관련 280 개<br>- 기타 420 개 | [sft_dataset_llm.csv](../create_dataset/sft_dataset_llm.csv)   |
| Odd-Radio Preference Optimizaiton (ORPO) | **총 199 개 규모** 데이터셋                                       | [orpo_dataset_llm.csv](../create_dataset/orpo_dataset_llm.csv) |

## 2. Supervised Fine-Tuning (SFT)

**1. Fine-Tuning 방법**

* [Supervised Fine-Tuning (SFT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)
  * 지도학습을 통해 모델이 diagram format 에 맞는 text 를 생성하도록 유도
  * [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md#2-lora-low-rank-adaptation) 를 선택
* [Odd-Radio Preference Optimizaiton (ORPO)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_DPO_ORPO.md#3-orpo-odds-ratio-preference-optimization)
  * 유저가 선호하는 스타일의 diagram 이 생성되는 text 를 생성하도록 유도 
* 각 방법의 선택 이유

| 방법         | 선택 이유                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SFT + LoRA | - 특정 포맷대로 생성하는 LLM에서 **해당 포맷과의 오차를 최대한 줄여서 일치시키는** 것이 중요하므로, SFT (지도 학습) 선택<br>- 빠른 학습을 위해 [PEFT](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md) 방법 사용<br>- LoRA 는 LLM Fine-tuning 의 최신 트렌드인 만큼 장점이 많을 것으로 판단<br>- 다른 방법인 [Prefix Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-3-prefix-tuning) / [Prompt Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-4-prompt-tuning) / [Adapter Layer 추가](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-5-adapter-layer-%EC%B6%94%EA%B0%80) 는 **여러 task 를 처리하는 LLM** 에 보다 적합한데, 본 프로젝트는 단일 task 임 |
| ORPO       | - [RLHF](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_DPO_ORPO.md#1-1-rlhf-reinforcement-learning-from-human-feedback) 와 같은 **강화학습을 사용하지 않고도 사용자가 선호하는 답변** 생성<br>- [DPO](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_DPO_ORPO.md#2-dpo-direct-preference-optimization) 는 참조 모델이 있어야 하므로 GPU 메모리 측면에서 부담될 수 있음                                                                                                                                                                                                                                                                                                                                                                                                                      |

**2. 데이터셋**

* ```create_dataset/sft_dataset_llm.csv```
* 총 700 개 규모 데이터셋
  * 딥러닝 모델 구조 관련 280 개
  * 기타 420 개

**3. Fine-Tuning 코드**

* 학습 코드
  * ```fine_tuning/sft_fine_tuning.py```
* SFT 된 모델을 이용한 ORPO 용 데이터 생성 코드
  * ```fine_tuning/orpo_create_dataset.py```
  * 해당 코드 실행 시, ```create_dataset/orpo_dataset_llm.csv``` 에 ORPO 용 추가 데이터를 생성하여 추가
* 학습 설정
  * training batch size = 1
  * gradient checkpointing 적용
  * [학습 결과](log/log_train_final_sft.md) (학습 종료 시점에서 **평균 training loss 0.075** 내외)

**4. Fine-Tuning 결과**

* 결과 요약
  * 데이터 포맷에 대한 Fine-tuning 자체는 제대로 된 듯함 
    * [최초의 deepseek-coder-1.3b-instruct LLM](../test_llm/README.md/#3-테스트-진행-및-결과) 이 형식에 맞게 응답한 것이 **13 / 20 개** 였는데 비해, [ORPO 데이터셋 생성용으로 작성한 프롬프트 200개](../create_dataset/orpo_dataset_llm.csv) 를 이용하여 테스트 결과 **(일단 형식을 맞춘 것은) 199 / 200 개** 임
  * 실제 유저가 원하는 다이어그램 생성은 미흡함
    * 신경망, CNN 모델, Flow-Chart 다이어그램 중 일부는 의도한 다이어그램 형태에 상당히 근접함
    * 그러나, 도형이 겹치거나 canvas 범위를 넘어가는 등 가독성이 떨어지는 부분이 있음

* 상세 이미지
  * 총 다이어그램 200 개
  * 해당 이미지 생성을 위한 SFT-applied LLM 의 answer 는 이후 ORPO 의 학습 데이터 중 rejected sample 로 들어감

**1 ~ 90 번째 다이어그램**

![image](../../images/250312_12.PNG)

**91 ~ 150 번째 다이어그램**

![image](../../images/250312_13.PNG)

**151 ~ 200 번째 다이어그램**

![image](../../images/250312_14.PNG)

## 3. Odd-Radio Preference Optimizaiton (ORPO) (실패, 미 적용)

**1. 데이터셋**

* ```create_dataset/orpo_dataset_llm.csv```
  * **SFT 된 모델로 ORPO 용 추가 데이터를 생성한 후** 의 csv 파일이어야 함
* 총 199 개 규모 데이터셋을 바탕으로 ORPO 학습 데이터셋 생성
  * 원래 LLM 이 목표로 하는 output 을 chosen 으로 간주
  * SFT 로 Fine-Tuning 된 모델을 이용하여 생성한 output 을 rejected 로 간주
  * 200개 데이터 중 1개는 SFT 로 Fine-Tuning 된 모델이 출력한 output 에 대한 점수가 만점이므로, rejected 로 간주 불가하여 데이터셋에서 제외

**2. Fine-Tuning 코드**

* 학습 코드
  * ```fine_tuning/orpo_fine_tuning.py```

**3. Fine-Tuning 결과**

* [Cuda OOM 으로 인한 학습 실패](../README.md#5-8-orpo-학습-시-cuda-out-of-memory-해결-실패)
* 환경적 제약에 의한 문제로, 본 프로젝트 일정 내에는 극복 불가 판단

## 4. 코드 실행 순서

* 준비 사항
  * ```create_dataset/sft_dataset_llm.csv```
  * ```create_dataset/orpo_dataset_llm.csv``` 의 중간 버전 (SFT format 의 score = 1.0 인 데이터만 존재)

* 실행 순서

```commandline
python sft_fine_tuning.py
python orpo_create_dataset.py
python orpo_fine_tuning.py
```