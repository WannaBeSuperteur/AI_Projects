## 목차

* [1. LLM Final Selection](#1-llm-final-selection)
  * [1-1. Polyglot-Ko 1.3B 선택 이유](#1-1-polyglot-ko-13b-선택-이유) 
  * [1-2. 참고: Gemma License](#1-2-참고-gemma-license)
* [2. How to run Fine-Tuning](#2-how-to-run-fine-tuning)
* [3. LLM Memory (RAG-like concept)](#3-llm-memory-rag-like-concept)
* [4. Test / Run Model](#4-test--run-model)
  * [4-1. Prepare Model (Gemma-2 2B Based)](#4-1-prepare-model-gemma-2-2b-based)
  * [4-2. Prepare Model (Polyglot-Ko 1.3B Based)](#4-2-prepare-model-polyglot-ko-13b-based)
  * [4-3. Unsloth use test](#4-3-unsloth-use-test)
  * [4-4. Run LLM Fine-Tuning](#4-4-run-llm-fine-tuning)
  * [4-5. Run Final Fine-Tuned Model](#4-5-run-final-fine-tuned-model)

## 1. LLM Final Selection

* **Polyglot-Ko 1.3B (1.43 B params)**
  * [HuggingFace](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
* [LLM Selection Report](model_selection/README.md) 기준
  * 최종 모델 : **Gemma-2 2B**
  * 예비 모델 : **Polyglot-Ko 1.3B (✅ 최종 채택)**

### 1-1. Polyglot-Ko 1.3B 선택 이유

* **실제 Fine-Tuning 된 모델** 의 생성 문장 측면 [(Gemma-2 2B 테스트 결과)](fine_tuning/fine_tuning_logs/2504221644%20(Inference,%2025042213%20dataset,%20temp=1.2).txt) [(Polyglot-Ko 1.3B 테스트 결과)](fine_tuning/fine_tuning_logs_polyglot/2504230855%20(Inference,%20epochs=60,%20rank=64,%20temp=0.6).txt)
  * Gemma-2 2B 가 Polyglot-Ko 1.3B 보다 생성 문장의 품질이 전반적으로 떨어짐
    * Gemma-2 2B 는 동일 질문에 대해 **유사한 답변을 생성** 하는 빈도가 Polyglot-Ko 1.3B 보다 현저히 높음
    * Gemma-2 2B 는 **특정 질문에 대해 empty answer 를 생성 (읽씹)** 하는 빈도가 Polyglot-Ko 1.3B 보다 현저히 높음
    * Gemma-2 2B 는 **어색한 외국어 문장** 을 생성하는 경우가 많음 
  * memory 정보 (예: ```[오늘 일정: 신규 아이템 발표]```) 파악 및 패드립 대응 (경고 처리) 능력은 Gemma-2 2B 가 Polyglot-Ko 1.3B 보다 높은 편이지만, 치명적인 이슈는 아님
* 기타
  * Gemma-2 2B 는 Polyglot-Ko 1.3B 와 달리 [Totally Free License 가 아님](#1-2-참고-gemma-license)
  * Polyglot-Ko 1.3B 는 **파라미터 개수가 Gemma-2 2B 의 절반 수준 (2.61 B vs. 1.43 B)**
    * 즉, 학습/추론 시간 및 메모리 사용량 관점에서 비교적 가볍고 빠름

### 1-2. 참고: Gemma License

* Source : [Gemma Terms of Use > Use Restrictions](https://ai.google.dev/gemma/terms#3.2-use)
* Checked Date : Apr 21, 2025 (KST)

----

3.2 Use Restrictions

You must not use any of the Gemma Services:

* for the restricted uses set forth in the Gemma Prohibited Use Policy at [ai.google.dev/gemma/prohibited_use_policy](https://ai.google.dev/gemma/prohibited_use_policy) **("Prohibited Use Policy")**, which is hereby incorporated by reference into this Agreement; or
* in violation of applicable laws and regulations.

To the maximum extent permitted by law, Google reserves the right to restrict (remotely or otherwise) usage of any of the Gemma Services that Google reasonably believes are in violation of this Agreement.

----

## 2. How to run Fine-Tuning

**1. Fine-Tuning 방법 및 데이터셋**

* 학습 모델
  * **Polyglot-Ko 1.3B (1.43 B params)** [HuggingFace](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) 
* 학습 방법 
  * [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)
  * [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md), LoRA Rank = 64
  * train for **60 epochs (= 2h 21m)** [(train report)](fine_tuning/fine_tuning_logs_polyglot/2504230200%20(stop%20crit,%2060%20epochs,%20rank=64).txt)
* 학습 데이터셋
  * [Train & Valid Dataset](OhLoRA_fine_tuning_25042213.csv) (**360** Q & A pairs for training / **60** Q & A pairs for validation) 
* Fine-Tuning 방법 선택 근거
  * 메모리 및 연산량을 절약 가능한, 최근 많이 쓰이는 LLM Fine-Tuning 방법 중 하나
  * **Oh-LoRA (오로라)** 라는 이름의 상징성을 고려
  * 널리 알려진 다른 방법들인 [Prefix Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-3-prefix-tuning), [Prompt Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-4-prompt-tuning), [Adapter Layer 추가](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-5-adapter-layer-%EC%B6%94%EA%B0%80) 등은 Multi-task LLM 에 보다 적합한데, 본 LLM 은 **단순 대화형 LLM 이 목적이므로 Multi-task 로 보기 다소 어려움**

**2. 모델 별 학습 특이 사항**

* **Polyglot-Ko 1.3B**
  * 학습 시 문제점 
    * Original Polyglot-Ko 1.3B LLM 의 tokenizer 의 end-of-sequence token 인 ```<|endoftext|>``` 가, **Fine-Tuning 된 모델에서는 매우 드물게 생성** 됨
    * 이로 인해 **거의 대부분의 생성 문장이 max token length 인 80 에 도달함**
  * 해결책으로, 다음 방법을 이용
    * 모든 학습 데이터의 답변 부분의 끝에 ```(답변 종료)``` 문구를 추가 후,
    * ```stopping_criteria``` 를 이용하여 ```(답변 종료)``` 에 해당하는 token 이 출력될 시 문장 생성 중지

## 3. LLM Memory (RAG-like concept)

* TBU

## 4. Test / Run Model

### 4-1. Prepare Model (Gemma-2 2B Based)

**1. Gemma-2-2b Original Unsloth Model (by Google & Unsloth)**

* ```2025_04_08_OhLoRA/llm/models/original``` 에, [gemma-2-2b-it Hugging-Face](https://huggingface.co/unsloth/gemma-2-2b-it/tree/main) 에서 다운로드받은 모델 및 관련 파일 저장
  * 총 9 개 파일 (각종 정보 포함)
  * 이때 ```models/original``` 디렉토리는 Clone 받은 repo. 에 원래 없으므로, 새로 생성  

* **🚨 VERY IMPORTANT 🚨**
  * **추가 개발 또는 실제 사용 시, [Gemma License](#1-2-참고-gemma-license) 를 준수해야 함**

```
2025_04_08_OhLoRA
- llm
  - fine_tuning
  - model_selection
  - models                        (필요 시 디렉토리 새로 생성)
    - original                    (디렉토리 새로 생성)
      - .gitarrtibutes            (다운로드 받은 파일)
      - README.md                 (다운로드 받은 파일)
      - config.json               (다운로드 받은 파일)
      - generation_config.json    (다운로드 받은 파일)
      - model.safetensors         (다운로드 받은 파일)
      - special_tokens_map.json   (다운로드 받은 파일)
      - tokenizer.json            (다운로드 받은 파일)
      - tokenizer.model           (다운로드 받은 파일)
      - tokenizer_config.json     (다운로드 받은 파일)
    - ...
  - unsloth_test
  - ...  
```

**2. Final Fine-Tuned Model**

* ```2025_04_08_OhLoRA/llm/models/fine_tuned``` 에 모델 저장
* TBU (기존 모델 준비 방법)

### 4-2. Prepare Model (Polyglot-Ko 1.3B Based)

**1. Polyglot-Ko Original Model (by EleutherAI)**

* ```2025_04_08_OhLoRA/llm/models/polyglot_original``` 에, [Polyglot-Ko 1.3B Hugging-Face](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) 에서 다운로드받은 모델 및 관련 파일 저장
  * 총 12 개 파일 (각종 정보 포함)
  * 이때 ```models/polyglot_original``` 디렉토리는 Clone 받은 repo. 에 원래 없으므로, 새로 생성

```
2025_04_08_OhLoRA
- llm
  - fine_tuning
  - model_selection
  - models                                 (필요 시 디렉토리 새로 생성)
    - polyglot_original                    (디렉토리 새로 생성)
      - .gitarrtibutes                     (다운로드 받은 파일)
      - README.md                          (다운로드 받은 파일)
      - config.json                        (다운로드 받은 파일)
      - generation_config.json             (다운로드 받은 파일)
      - model-00001-of-00003.safetensors   (다운로드 받은 파일)
      - model-00002-of-00003.safetensors   (다운로드 받은 파일)
      - model-00003-of-00003.safetensors   (다운로드 받은 파일)
      - model.safetensors.index.json       (다운로드 받은 파일)
      - pytorch_model.bin                  (다운로드 받은 파일)
      - special_tokens_map.json            (다운로드 받은 파일)
      - tokenizer.json                     (다운로드 받은 파일)
      - tokenizer_config.json              (다운로드 받은 파일)
    - ...
  - unsloth_test
  - ...  
```

**2. Final Fine-Tuned Model**

* ```2025_04_08_OhLoRA/llm/models/polyglot_fine_tuned``` 에 모델 저장
* TBU (기존 모델 준비 방법)

### 4-3. Unsloth use test

**1. 실험 목적**

* 2024년에 공개된 [Unsloth](https://unsloth.ai/) 라는 툴을 이용하면 **LLM 의 학습 시간 및 메모리 등 자원을 절약** 할 수 있다.
* 본 task (with Quadro 6000 12GB GPU) 에서는 어느 정도의 속도 향상 및 메모리 절감이 있는지 파악하여 향후 참고한다.

**2. 실험 결과**

* 결론
  * **❌ Quadro M6000 (12 GB) 에서 Unsloth 학습 불가능** [(참고)](https://github.com/unslothai/unsloth/issues/1998)

```
RuntimeError: Found Quadro M6000 which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 5.2       
```

| 테스트              | Inference 메모리<br>(nvidia-smi 측정값) | Inference 시간 | Fine-Tuning 메모리<br>(nvidia-smi 측정값) | Fine-Tuning 시간 |
|------------------|-----------------------------------|--------------|-------------------------------------|----------------|
| **with** Unsloth | 2,169 MB (3,829 MB) ❌             | ❌            | ❌                                   | ❌              |
| **w/o** Unsloth  | 5,013 MB (5,924 MB)               | 158.2 s      | 5,994 MB (9,503 MB)                 | 19.7 s         |
| 절감               | 56.7 % (35.4 %)                   | -            | -                                   | -              |

**3. 실험 설정**

* Inference
  * run inference on 25 data (Q & A pairs)
* Fine-Tuning
  * **60 (data * epochs) = 20 data * 3 epochs**
    * 실제 학습 데이터의 **16.7% = 0.167 epochs** 분량
  * epoch & batch size
    * 3 epochs
    * train batch size = 4
    * valid batch size = 1
  * data
    * total 20 data (Q & A pairs) for training
    * 5 data (Q & A pairs) for validation

**3. 코드 실행 방법**

* 먼저, [Prepare Model](#4-1-prepare-model) 에 나온 대로 모델 준비
* ```2025_04_08_OhLoRA``` 메인 디렉토리에서 실행

| 테스트              | Python 명령어                                            |
|------------------|-------------------------------------------------------|
| **with** Unsloth | ```python llm/unsloth_test/test_with_unsloth.py```    |
| **w/o** Unsloth  | ```python llm/unsloth_test/test_without_unsloth.py``` |

### 4-4. Run LLM Fine-Tuning

* Gemma2 2B Fine Tuning
  * 프로젝트 메인 디렉토리 (```2025_04_08_OhLoRA```) 에서 ```python llm/run_fine_tuning.py``` 실행
* **Polyglot-Ko 1.3B (✅ 최종 채택)** Fine Tuning
  * 프로젝트 메인 디렉토리 (```2025_04_08_OhLoRA```) 에서 ```python llm/run_fine_tuning_polyglot.py``` 실행

### 4-5. Run Final Fine-Tuned Model

* 먼저 (TBU) 를 참고하여 **모델 (Polyglot-Ko 1.3B Fine-Tuned LLM)** 준비
* TBU