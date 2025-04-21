## 목차

* [1. LLM Final Selection](#1-llm-final-selection)
  * [1-1. Gemma License](#1-1-gemma-license) 
* [2. How to run Fine-Tuning](#2-how-to-run-fine-tuning)
* [3. LLM Memory (RAG-like concept)](#3-llm-memory-rag-like-concept)

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