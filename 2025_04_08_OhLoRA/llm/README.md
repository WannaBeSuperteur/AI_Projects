## 목차

* [1. LLM Final Selection](#1-llm-final-selection)
  * [1-1. Gemma License](#1-1-gemma-license) 
* [2. How to run Fine-Tuning](#2-how-to-run-fine-tuning)
* [3. LLM Memory (RAG-like concept)](#3-llm-memory-rag-like-concept)
* [4. Test / Run Model](#4-test--run-model)
  * [4-1. Prepare Model](#4-1-prepare-model)
  * [4-2. Unsloth use test](#4-2-unsloth-use-test)
  * [4-3. Run Final Fine-Tuned Model](#4-3-run-final-fine-tuned-model)

## 1. LLM Final Selection

* **gemma-2-2b-it (2.61 B params)**
  * [HuggingFace](https://huggingface.co/unsloth/gemma-2-2b-it)
  * [License : gemma](https://ai.google.dev/gemma/terms) **(NOT totally free !!)**
* [LLM Selection Report](model_selection/README.md)

### 1-1. Gemma License

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

* Fine-Tuning 방법
  * [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)
    * [Train Dataset](OhLoRA_fine_tuning.csv) (200 Q & A pairs for training)
  * [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md)
    * LoRA Rank = 64
* Fine-Tuning 방법 선택 근거
  * 메모리 및 연산량을 절약 가능한, 최근 많이 쓰이는 LLM Fine-Tuning 방법 중 하나
  * **Oh-LoRA (오로라)** 라는 이름의 상징성을 고려
  * 널리 알려진 다른 방법들인 [Prefix Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-3-prefix-tuning), [Prompt Tuning](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-4-prompt-tuning), [Adapter Layer 추가](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_PEFT.md#2-5-adapter-layer-%EC%B6%94%EA%B0%80) 등은 Multi-task LLM 에 보다 적합한데, 본 LLM 은 **단순 대화형 LLM 이 목적이므로 Multi-task 로 보기 다소 어려움**

## 3. LLM Memory (RAG-like concept)

* TBU

## 4. Test / Run Model

### 4-1. Prepare Model

**1. Gemma-2-2b Original Unsloth Model (by Google & Unsloth)**

* ```2025_04_08_OhLoRA/llm/models/original``` 에, [gemma-2-2b-it Hugging-Face](https://huggingface.co/unsloth/gemma-2-2b-it/tree/main) 에서 다운로드받은 모델 및 관련 파일 저장
  * 총 9 개 파일 (각종 정보 포함)
  * 이때 ```models/original``` 디렉토리는 Clone 받은 repo. 에 원래 없으므로, 새로 생성  

* **추가 개발 또는 실제 사용 시, [Gemma License](#1-1-gemma-license) 를 준수해야 함**

```
2025_04_08_OhLoRA
- llm
  - model_selection
  - models (디렉토리 새로 생성)
    - original (디렉토리 새로 생성)
      - .gitarrtibutes (다운로드)
      - README.md (다운로드)
      - config.json (다운로드)
      - generation_config.json (다운로드)
      - model.safetensors (다운로드)
      - special_tokens_map.json (다운로드)
      - tokenizer.json (다운로드)
      - tokenizer.model (다운로드)
      - tokenizer_config.json (다운로드)
  - unsloth_test
  - ...  
```

**2. Final Fine-Tuned Model for Oh-LoRA (오로라)**

TBU

### 4-2. Unsloth use test

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
    * 실제 학습 데이터의 **30% = 0.3 epochs** 분량
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

### 4-3. Run Final Fine-Tuned Model

* TBU