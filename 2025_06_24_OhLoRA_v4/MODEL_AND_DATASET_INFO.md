## 목차

* [1. 모델 정보](#1-모델-정보)
* [2. 모델 배치 방법](#2-모델-배치-방법)
  * [2-1. output_message LLM 파일](#2-1-output_message-llm-파일)
  * [2-2. memory LLM 파일](#2-2-memory-llm-파일)
  * [2-3. summary LLM 파일](#2-3-summary-llm-파일)
  * [2-4. eyes_mouth_pose LLM 파일](#2-4-eyes_mouth_pose-llm-파일)
  * [2-5. S-BERT 모델 파일 (Memory)](#2-5-s-bert-모델-파일-memory)
  * [2-6. S-BERT 모델 파일 (Ethics)](#2-6-s-bert-모델-파일-ethics)
* [3. 데이터셋 정보](#3-데이터셋-정보)

## 1. 모델 정보

* 아래 표에 표시된 모든 모델 **(총 8개)** 은 **Oh-LoRA 👱‍♀️ (오로라) v4** 의 원활한 동작을 위해 필요
* 본 문서에 표시된 모든 저장 경로는 **처음부터 ```/2025_06_24_OhLoRA_v4/``` 까지의 경로는 제외하고 나타낸** 경로임.
* ```stylegan/models``` 디렉토리가 처음 clone 받았을 때는 없으므로, **해당 디렉토리를 먼저 생성 후** 진행

| 모델                                                                                                                                                         | 용도                                                                                                                                                                                                                                                              | 다운로드 경로                                                                                                                         | 모델 파일 배치 방법                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [StyleGAN-VectorFind-v7](../2025_05_02_OhLoRA_v2/stylegan/README.md#3-3-stylegan-finetune-v1-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v7) | Oh-LoRA 👱‍♀️ (오로라) **얼굴 이미지 생성**                                                                                                                                                                                                                               | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_StyleGAN_VectorFind)<br> > ```stylegan_gen_vector_find_v7.pth``` | ```stylegan/models``` 경로에 원래 이름 그대로 저장     |
| [StyleGAN-VectorFind-v8](../2025_05_26_OhLoRA_v3/stylegan/README.md#3-3-stylegan-finetune-v8-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v8) | Oh-LoRA 👱‍♀️ (오로라) **얼굴 이미지 생성**                                                                                                                                                                                                                               | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_StyleGAN_VectorFind)<br> > ```stylegan_gen_vector_find_v8.pth``` | ```stylegan/models``` 경로에 원래 이름 그대로 저장     |
| Oh-LoRA LLM<br>**(output_message)**                                                                                                                        | Oh-LoRA 👱‍♀️ (오로라) 의 **LLM 답변 출력**                                                                                                                                                                                                                             | [HuggingFace](https://huggingface.co/daebakgazua/250624_OhLoRA_LLM_output_message)                                              | [모델 파일 배치 방법](#2-1-output_message-llm-파일)  |
| Oh-LoRA LLM<br>**(memory)**                                                                                                                                | Oh-LoRA 👱‍♀️ (오로라) 가 **자체 메모리에 저장** 해야 할, **기억해야 하는 정보** 추출 ([참고](../2025_05_26_OhLoRA_v3/llm/README.md#1-2-llm-memory-rag-like-concept))                                                                                                                      | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_LLM_polyglot/tree/main/memory)                                   | [모델 파일 배치 방법](#2-2-memory-llm-파일)          |
| Oh-LoRA LLM<br>**(summary)**                                                                                                                               | Oh-LoRA 👱‍♀️ (오로라) 와의 대화의 **가장 최근 turn 요약 → 다음 turn 에서 해당 정보 활용**                                                                                                                                                                                              | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_LLM_kanana/tree/main/summary)                                    | [모델 파일 배치 방법](#2-3-summary-llm-파일)         |
| Oh-LoRA LLM<br>**(eyes_mouth_pose)**                                                                                                                       | Oh-LoRA 👱‍♀️ (오로라) 얼굴 이미지 생성을 위한 **표정 정보** ([핵심 속성 값](../2025_05_26_OhLoRA_v3/stylegan/README.md#2-핵심-속성-값)) 를 **자연어로** 출력                                                                                                                                     | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_LLM_polyglot/tree/main/eyes_mouth_pose)                          | [모델 파일 배치 방법](#2-4-eyes_mouth_pose-llm-파일) |
| S-BERT Model (for [Oh-LoRA Memory Concept](llm/README.md#1-ohlora-v4-llm-전체-메커니즘))                                                                         | Oh-LoRA 👱‍♀️ (오로라) 가 자체 메모리에서 **현재 대화에 맞는 적절한 정보** 를 유사도 기반으로 가져오기 위한 모델<br>- 참고 : [S-BERT 란?](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_LLM_SBERT/tree/main/memory_sbert/trained_sbert_model)            | [모델 파일 배치 방법](#2-5-s-bert-모델-파일-memory)    |
| S-BERT Model (for [Oh-LoRA Ethics Concept](llm/README.md#1-1-llm-ethics-s-bert))                                                                           | Oh-LoRA 👱‍♀️ (오로라) 가 사용자의 발언이 **윤리적으로 부적절한 발언은 아닌지** 판단하기 위한 모델<br>- 참고 : [S-BERT 란?](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8)          | [HuggingFace](https://huggingface.co/daebakgazua/250624_OhLoRA_LLM_output_message/tree/main/ethics_sbert/trained_sbert_model)   | [모델 파일 배치 방법](#2-6-s-bert-모델-파일-ethics)    |

## 2. 모델 배치 방법

### 2-1. output_message LLM 파일

```llm/models/kananai_output_message_fine_tuned``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_06_24_OhLoRA_v4
- llm
  - models                                  (필요 시 디렉토리 새로 생성)
    - kananai_output_message_fine_tuned      (디렉토리 새로 생성)
      - adapter_config.json                 (다운로드 받은 파일)
      - adapter_model.safetensors           (다운로드 받은 파일)
      - special_tokens_map.json             (다운로드 받은 파일)
      - tokenizer.json                      (다운로드 받은 파일)
      - tokenizer_config.json               (다운로드 받은 파일)
      - training_args.bin                   (다운로드 받은 파일)
    - ...
  - ...
```

### 2-2. memory LLM 파일

```llm/models/polyglot_memory_fine_tuned``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_06_24_OhLoRA_v4
- llm
  - models                                  (필요 시 디렉토리 새로 생성)
    - polyglot_memory_fine_tuned            (디렉토리 새로 생성)
      - adapter_config.json                 (다운로드 받은 파일)
      - adapter_model.safetensors           (다운로드 받은 파일)
      - special_tokens_map.json             (다운로드 받은 파일)
      - tokenizer.json                      (다운로드 받은 파일)
      - tokenizer_config.json               (다운로드 받은 파일)
      - training_args.bin                   (다운로드 받은 파일)
    - ...
  - ...
```

### 2-3. summary LLM 파일

```llm/models/kanana_summary_fine_tuned``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_06_24_OhLoRA_v4
- llm
  - models                                  (필요 시 디렉토리 새로 생성)
    - kanana_summary_fine_tuned             (디렉토리 새로 생성)
      - adapter_config.json                 (다운로드 받은 파일)
      - adapter_model.safetensors           (다운로드 받은 파일)
      - special_tokens_map.json             (다운로드 받은 파일)
      - tokenizer.json                      (다운로드 받은 파일)
      - tokenizer_config.json               (다운로드 받은 파일)
      - training_args.bin                   (다운로드 받은 파일)
    - ...
  - ...
```

### 2-4. eyes_mouth_pose LLM 파일

```llm/models/polyglot_eyes_mouth_pose_fine_tuned``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_06_24_OhLoRA_v4
- llm
  - models                                  (필요 시 디렉토리 새로 생성)
    - polyglot_eyes_mouth_pose_fine_tuned   (디렉토리 새로 생성)
      - adapter_config.json                 (다운로드 받은 파일)
      - adapter_model.safetensors           (다운로드 받은 파일)
      - special_tokens_map.json             (다운로드 받은 파일)
      - tokenizer.json                      (다운로드 받은 파일)
      - tokenizer_config.json               (다운로드 받은 파일)
      - training_args.bin                   (다운로드 받은 파일)
    - ...
  - ...
```

### 2-5. S-BERT 모델 파일 (Memory)

```llm/models/memory_sbert/trained_sbert_model``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_06_24_OhLoRA_v4
- llm
  - models                                                     (필요 시 디렉토리 새로 생성)
    - memory_sbert                                             (디렉토리 새로 생성)
      - trained_sbert_model                                    (디렉토리 새로 생성)
        - 1_Pooling                                            (디렉토리 새로 생성)
          - config.json                                        (다운로드 받은 파일)
        - eval                                                 (디렉토리 새로 생성)
          - similarity_evaluation_valid_evaluator_results.csv  (다운로드 받은 파일)
        - config.json                                          (다운로드 받은 파일)
        - config_sentence_transformers.json                    (다운로드 받은 파일)
        - model.safetensors                                    (다운로드 받은 파일)
        - modules.json                                         (다운로드 받은 파일)
        - sentence_bert_config.json                            (다운로드 받은 파일)
        - special_tokens_map.json                              (다운로드 받은 파일)
        - tokenizer.json                                       (다운로드 받은 파일)
        - tokenizer_config.json                                (다운로드 받은 파일)
        - vocab.txt                                            (다운로드 받은 파일)
    - ...
  - ...
```

### 2-6. S-BERT 모델 파일 (Ethics)

```llm/models/ethics_sbert/trained_sbert_model``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_06_24_OhLoRA_v4
- llm
  - models                                                     (필요 시 디렉토리 새로 생성)
    - ethics_sbert                                             (디렉토리 새로 생성)
      - trained_sbert_model                                    (디렉토리 새로 생성)
        - 1_Pooling                                            (디렉토리 새로 생성)
          - config.json                                        (다운로드 받은 파일)
        - eval                                                 (디렉토리 새로 생성)
          - similarity_evaluation_valid_evaluator_results.csv  (다운로드 받은 파일)
        - config.json                                          (다운로드 받은 파일)
        - config_sentence_transformers.json                    (다운로드 받은 파일)
        - model.safetensors                                    (다운로드 받은 파일)
        - modules.json                                         (다운로드 받은 파일)
        - sentence_bert_config.json                            (다운로드 받은 파일)
        - special_tokens_map.json                              (다운로드 받은 파일)
        - tokenizer.json                                       (다운로드 받은 파일)
        - tokenizer_config.json                                (다운로드 받은 파일)
        - vocab.txt                                            (다운로드 받은 파일)
    - ...
  - ...
```

## 3. 데이터셋 정보

* [EffiSegNet 기반 Oh-LoRA v4 용 Segmentation 모델](segmentation/README.md) 의 학습 데이터
* 총 4,930 장의 **hair area segmentation result**
* [HuggingFace Link](https://huggingface.co/datasets/daebakgazua/250624_OhLoRA_Hair_Segmentation_Result)
