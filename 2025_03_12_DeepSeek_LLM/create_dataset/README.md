
## 1. ```sft_dataset.py``` 설명

* [Supervised Fine-Tuning (SFT)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md) 용 데이터셋 생성 코드
* 총 100 개 규모 데이터셋
  * 딥러닝 모델 구조 관련 40 개
  * 기타 60 개

## 2. ```orpo_dataset.py``` 설명

* [ORPO (Odd-Ratio Preference Optimization)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_DPO_ORPO.md#3-orpo-odds-ratio-preference-optimization) 용 데이터셋 생성 코드
* 총 300 개 규모 데이터셋
  * 딥러닝 모델 구조 관련 120 개
  * 기타 180 개

## 3. ```common.py``` 설명

* SFT, ORPO 에 공통으로 사용되는 형태의 데이터셋 생성의 경우, 관련 함수를 common.py 에 정의