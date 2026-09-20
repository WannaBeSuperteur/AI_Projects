
# Text Embedding Model 후보 목록

## 목차

* [1. 요약](#1-요약)
* [2. 리더보드 및 벤치마크 선택](#2-리더보드-및-벤치마크-선택)
* [3. 최종 모델 선정](#3-최종-모델-선정)
  * [3-1. `CodeTransOceanContest` 벤치마크에 대해 경량 모델 재 선정](#3-1-codetransoceancontest-벤치마크에-대해-경량-모델-재-선정) 

## 1. 요약

**2026.09.17 (목) 모델 재 선정 기준**

* 최종 선정 벤치마크
  * `CodeSearchNetCCRetrieval` (1개 규칙)
  * `CodeTransOceanContest` (9개 규칙)
  * `COIRCodeSearchNetRetrieval` (2개 규칙)
* 최종 선정 모델
  * `gte-modernbert-base`
  * `F2LLM-v2-330M`
  * `LateOn-Code-pretrain` (`Giga-Embeddings-instruct-480M-0826` 대체 경량화 모델) [(참고)](#3-1-codetransoceancontest-벤치마크에-대해-경량-모델-재-선정)

## 2. 리더보드 및 벤치마크 선택

* 리더보드
  * **선택: [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)**
  * 참고: [S-BERT Benchmark](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) 는 2026년 텍스트 임베딩의 최신 트렌드와 거리가 있다고 판단됨

* 벤치마크 데이터셋
  * Task (Python 코드 검토) 에 따른 후보 선택
  * 조사 기준일: **2026.09.17 (목)**

| 벤치마크 데이터셋                                                                             | 모델 개수<br>(500M 이하 / 200M 이하) | task 개수 | Python 포함 여부 |
|---------------------------------------------------------------------------------------|------------------------------|---------|--------------|
| [Code Information Retrieval (CoIR)](https://mteb-leaderboard.hf.space/benchmark/CoIR) | 56 **(31 / 22)**             | 10      | ✅            |
| [Code](https://mteb-leaderboard.hf.space/benchmark/MTEB(Code%2C%20v1))                | 48 **(27 / 18)**             | 12      | ✅            |
| [CoREB (v1)](https://mteb-leaderboard.hf.space/benchmark/CoREB(v1))                   | 21 **(9 / 7)**               | 6       | ✅            |
| [RTEB Code](https://mteb-leaderboard.hf.space/benchmark/RTEB(Code%2C%20beta))         | 9 **(9 / 8)**                | 9       | ✅            |

* 최종 선정 벤치마크 (Task)
  * 위 표 기준으로, **500M 이하 모델 수** 가 많은 `CoIR`, `Code` 중심으로 선정
  * 🔄 : 비교적 무거운 `Giga-Embeddings-instruct-480M-0826` 모델 대신 **경량 모델로 재 선정** (선정 결과: `LateOn-Code-pretrain`) [(참고)](#3-1-codetransoceancontest-벤치마크에-대해-경량-모델-재-선정)
  * ➕ : **추가 요구사항** 에 의해 추가된 Oh-LoRA v7 규칙

| Oh-LoRA v7 규칙                             | 최종 선정 벤치마크                   | 선정 이유                                                                                                                                       |
|-------------------------------------------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 불필요한 print, logging 등이 없어야 함              | `CodeSearchNetCCRetrieval`   | code snippet 기준으로 판단해야 하므로, code snippet 최적화 벤치마크 선택                                                                                        |
| 🔄 유사한 변수명은 하나로 통일 시키는 것이 좋음              | `CodeTransOceanContest`      | - 변수명 유사도 판단 시 **자연어 맥락 고려 필요**<br>- Python 및 자연어의 2개 언어 역량을 동시에 평가<br>- 유사한 벤치마크인 `CodeSearchNetRetrieval` 보다 성능 측정값 누락 (missing value) 적음 |
| 🔄 변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것)       | `CodeTransOceanContest`      | - Python 및 자연어의 2개 언어 역량을 동시에 평가<br>- 유사한 벤치마크인 `CodeSearchNetRetrieval` 보다 성능 측정값 누락 (missing value) 적음                                    |
| 🔄 함수명과 반환값이 서로 잘 match 되어야 함             | `CodeTransOceanContest`      | 상동                                                                                                                                          |
| 함수의 단일 책임 원칙 준수 여부 (docstring 으로 판단)      | `COIRCodeSearchNetRetrieval` | code summary (docstring) 를 대상으로 하는 벤치마크임                                                                                                    |
| 함수 docstring과 함수명이 서로 일치하는지 판단            | `COIRCodeSearchNetRetrieval` | 상동                                                                                                                                          |
| 🔄 함수의 인자가 하나로 묶을 수 있는 경우 처리 필요           | `CodeTransOceanContest`      | `변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것)` 과 동일 **(자연어 맥락 고려 필요)**                                                                                  |
| 🔄 함수의 인자가 유동적인 경우 처리 필요                  | `CodeTransOceanContest`      | 상동                                                                                                                                          |
| 🔄 상태 값으로 판단되는 값을 조건으로 하는지 여부             | `CodeTransOceanContest`      | 상동                                                                                                                                          |
| 🔄 한 모듈 (*.py 파일) 내에서, 유사한 이름의 함수끼리 거리 검사 | `CodeTransOceanContest`      | 상동                                                                                                                                          |
| ➕ 숫자 값을 constant 처럼 사용 시, 해당 값 상수화 적절성 판단 | `CodeTransOceanContest`      | 상동                                                                                                                                          |
| ➕ 동일한 숫자 값 2회 이상 사용 시, 상수로 통합 적절성 판단      | `CodeTransOceanContest`      | 상동                                                                                                                                          |

## 3. 최종 모델 선정

* 선정 기준
  * 500M 이하 (메모리 부담 없이 신속한 코드 리뷰 가능한, 빠르게 추론 가능한 모델) 모델 필터링
    * 단, **경량 모델로 재선정** 대상의 경우 **200M 이하** 모델 필터링 
  * not zero-shot 측정 결과 제외 (⚠ 이모지 없음)<br>- not zero-shot 인 경우, **대부분의 모델이 zero-shot** 으로 비교가 어렵고, not zero-shot (예시 제공) 이 모델 성능에 유의미한 영향을 준다고 판단하여 선정에서 제외
  * 필터링 된 모델 중 `최종 선정 벤치마크` 기준 1위

* 선정 모델

| 최종 선정 벤치마크                          | 최종 선정 모델                                                                                                | 해당하는 Oh-LoRA v7 규칙                                                         |
|-------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `CodeSearchNetCCRetrieval`          | [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)                           | - 불필요한 print, logging 등이 없어야 함                                             |
| `COIRCodeSearchNetRetrieval`        | [F2LLM-v2-330M](https://huggingface.co/codefuse-ai/F2LLM-v2-330M)                                       | - 함수의 단일 책임 원칙 준수 여부 (docstring 으로 판단)<br>- 함수 docstring과 함수명이 서로 일치하는지 판단 |
| `CodeTransOceanContest`             | [Giga-Embeddings-instruct-480M-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826) | 없음 (모두 경량 모델로 교체)                                                          |
| `CodeTransOceanContest` (경량 모델 재선정) | [LateOn-Code-pretrain](https://huggingface.co/lightonai/LateOn-Code-pretrain)                           | 그 외 9개 (추가 요구사항 2개 포함)                                                     |

### 3-1. `CodeTransOceanContest` 벤치마크에 대해 경량 모델 재 선정

* 문제점
  * 기존 모델 `Giga-Embeddings-instruct-480M-0826` 는 **inference time 이 매우 긺**

* 발견 및 해결 방법
  * MAE, MSE가 낮은 4개 규칙 (아래 표 참고) 에 대해 경량 모델 `LateOn-Code-pretrain` 로 대체 실험 결과, MAE, MSE가 큰 차이 없음
  * 따라서, 경량 모델 `LateOn-Code-pretrain` 를 **해당 벤치마크를 적용한 7개 규칙 모두에 적용**

* 세부 사항
  * inference time : **250개** 테스트 데이터, **batch size = 4** 로 묶어서 테스트 기준

| Oh-LoRA v7 규칙                    | MAE<br>(모델 교체 전 / 교체 후)  | MSE<br>(모델 교체 전 / 교체 후)  | inference time (초)<br>(모델 교체 전 / 교체 후) |
|----------------------------------|--------------------------|--------------------------|----------------------------------------|
| 변수명, 함수명은 의미가 있어야 함 (+ 알기 쉽게 할것) | 0.0124 / 0.0117 (🔻)     | 0.0013 / 0.0014 (🔺)     | 26.222 / 8.843 (🔻)                    |
| 함수의 인자가 하나로 묶을 수 있는 경우 처리 필요     | 0.0258 / 0.0248 (🔻)     | 0.0051 / 0.0059 (🔺)     | 109.088 / 8.843 (🔻)                   |
| 함수의 인자가 유동적인 경우 처리 필요            | 0.0170 / 0.0173 (🔺)     | 0.0032 / 0.0034 (🔺)     | 109.784 / 8.906 (🔻)                   |
| 상태 값으로 판단되는 값을 조건으로 하는지 여부       | 0.0074 / 0.0095 (🔺)     | 0.0011 / 0.0015 (🔺)     | 109.184 / 8.796 (🔻)                   |
| **전체 평균**                        | **0.0157 / 0.0158 (🔺)** | **0.0027 / 0.0031 (🔺)** | **26 ~ 109 s / 8.8 s (🔻)**            |
