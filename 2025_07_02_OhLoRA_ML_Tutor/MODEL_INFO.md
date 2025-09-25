
## 목차

* [1. 모델 정보](#1-모델-정보)
* [2. 모델 배치 방법](#2-모델-배치-방법)
  * [2-1. ```qna``` LLM 파일](#2-1-qna-llm-파일)
  * [2-2. S-BERT 모델 파일 (```qna```)](#2-2-s-bert-모델-파일-qna)
  * [2-3. ```quiz``` LLM 파일](#2-3-quiz-llm-파일)
  * [2-4. S-BERT 모델 파일 (```quiz```)](#2-4-s-bert-모델-파일-quiz)
  * [2-5. ```interview``` LLM 파일](#2-5-interview-llm-파일)
  * [2-6. S-BERT 모델 파일 (```interview``` - 사용자 답변 성공 여부 파악)](#2-6-s-bert-모델-파일-interview---사용자-답변-성공-여부-파악)
  * [2-7. S-BERT 모델 파일 (```interview``` - 다음 질문 선택)](#2-7-s-bert-모델-파일-interview---다음-질문-선택)
  * [2-8. S-BERT 모델 파일 (Ethics)](#2-8-s-bert-모델-파일-ethics)

## 1. 모델 정보

* 아래 표에 표시된 모든 모델 **(총 11개)** 은 **Oh-LoRA 👱‍♀️ (오로라) ML Tutor** 의 원활한 동작을 위해 필요
* 본 문서에 표시된 모든 저장 경로는 **처음부터 ```/2025_07_02_OhLoRA_ML_Tutor/``` 까지의 경로는 제외하고 나타낸** 경로임.
* ```stylegan/models``` 및 ```ombre/models``` 디렉토리가 처음 clone 받았을 때는 없으므로, **해당 디렉토리를 먼저 생성 후** 진행

| 모델                                                                                                                                                         | 용도                                                                                                                                                                                                                                                     | 다운로드 경로                                                                                                                              | 모델 파일 배치 방법                                                  |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [StyleGAN-VectorFind-v7](../2025_05_02_OhLoRA_v2/stylegan/README.md#3-3-stylegan-finetune-v1-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v7) | Oh-LoRA 👱‍♀️ (오로라) **얼굴 이미지 생성**                                                                                                                                                                                                                      | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_StyleGAN_VectorFind)<br> > ```stylegan_gen_vector_find_v7.pth```      | ```stylegan/models``` 경로에 원래 이름 그대로 저장                       |
| [StyleGAN-VectorFind-v8](../2025_05_26_OhLoRA_v3/stylegan/README.md#3-3-stylegan-finetune-v8-기반-핵심-속성값-변환-intermediate-w-vector-탐색-stylegan-vectorfind-v8) | Oh-LoRA 👱‍♀️ (오로라) **얼굴 이미지 생성**                                                                                                                                                                                                                      | [HuggingFace](https://huggingface.co/daebakgazua/250526_OhLoRA_StyleGAN_VectorFind)<br> > ```stylegan_gen_vector_find_v8.pth```      | ```stylegan/models``` 경로에 원래 이름 그대로 저장                       |
| [Segmentation Model for Oh-LoRA v4](../2025_06_24_OhLoRA_v4/segmentation/README.md#1-segmentation-방법)                                                      | Oh-LoRA 👱‍♀️ (오로라) 얼굴에 **Ombre 헤어스타일 적용**                                                                                                                                                                                                             | [HuggingFace](https://huggingface.co/daebakgazua/250624_OhLoRA_Hair_Segmentation/tree/main) > ```segmentation_model_ohlora_v4.pth``` | ```segmentation_models``` 경로에 원래 이름 그대로 저장                   | 
| Oh-LoRA LLM<br>**(qna)**                                                                                                                                   | **ML 분야 질의응답** 기능에서 답변 생성                                                                                                                                                                                                                              | [HuggingFace](https://huggingface.co/daebakgazua/250702_OhLoRA_qna_llm/tree/main)                                                    | [모델 파일 배치 방법](#2-1-qna-llm-파일)                               |
| S-BERT Model<br>**(qna)**                                                                                                                                  | **ML 분야 질의응답** 기능에서 사용자의 질문 의도에 가장 맞는 정보를 DB에서 추출 (RAG 컨셉)                                                                                                                                                                                             | [HuggingFace](https://huggingface.co/daebakgazua/250702_OhLoRA_qna_sbert/tree/main)                                                  | [모델 파일 배치 방법](#2-2-s-bert-모델-파일-qna)                         |
| Oh-LoRA LLM<br>**(quiz)**                                                                                                                                  | **ML 분야 퀴즈** 기능에서 퀴즈 해설 생성                                                                                                                                                                                                                             | [HuggingFace](https://huggingface.co/daebakgazua/250702_OhLoRA_quiz_llm/tree/main)                                                   | [모델 파일 배치 방법](#2-3-quiz-llm-파일)                              |
| S-BERT Model<br>**(quiz)**                                                                                                                                 | **ML 분야 퀴즈** 기능에서 사용자 답안 채점                                                                                                                                                                                                                            | [HuggingFace](https://huggingface.co/daebakgazua/250702_OhLoRA_quiz_sbert/tree/main)                                                 | [모델 파일 배치 방법](#2-4-s-bert-모델-파일-quiz)                        |
| Oh-LoRA LLM<br>**(interview)**                                                                                                                             | **실전 면접** 기능에서 면접관 발화 생성                                                                                                                                                                                                                               | [HuggingFace](https://huggingface.co/daebakgazua/250702_OhLoRA_interview_llm/tree/main)                                              | [모델 파일 배치 방법](#2-5-interview-llm-파일)                         |
| S-BERT Model for output answer<br>**(interview)**                                                                                                          | **실전 면접** 기능에서 사용자가 성공한 답변이 무엇인지 판단                                                                                                                                                                                                                    | [HuggingFace](https://huggingface.co/daebakgazua/250702_OhLoRA_interview_sbert_output_answer/tree/main)                              | [모델 파일 배치 방법](#2-6-s-bert-모델-파일-interview---사용자-답변-성공-여부-파악) |
| S-BERT Model for next question<br>**(interview)**                                                                                                          | **실전 면접** 기능에서 면접관의 다음 질문 선택                                                                                                                                                                                                                           | [HuggingFace](https://huggingface.co/daebakgazua/250702_OhLoRA_interview_sbert_next_question/tree/main)                              | [모델 파일 배치 방법](#2-7-s-bert-모델-파일-interview---다음-질문-선택)        |
| S-BERT Model (for [Oh-LoRA Ethics Concept](../2025_06_24_OhLoRA_v4/llm/README.md#1-1-llm-ethics-s-bert))                                                   | Oh-LoRA 👱‍♀️ (오로라) 가 사용자의 발언이 **윤리적으로 부적절한 발언은 아닌지** 판단하기 위한 모델<br>- 참고 : [S-BERT 란?](https://github.com/WannaBeSuperteur/AI-study/blob/main/Natural%20Language%20Processing/Basics_BERT%2C%20SBERT%20%EB%AA%A8%EB%8D%B8.md#sbert-%EB%AA%A8%EB%8D%B8) | [HuggingFace](https://huggingface.co/daebakgazua/250624_OhLoRA_LLM_SBERT_Ethics/tree/main)                                           | [모델 파일 배치 방법](#2-8-s-bert-모델-파일-ethics)                      |

## 2. 모델 배치 방법

### 2-1. ```qna``` LLM 파일

```ai_qna/models/kananai_sft_final_fine_tuned``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- ai_qna
  - models                              (필요 시 디렉토리 새로 생성)
    - kananai_sft_final_fine_tuned      (디렉토리 새로 생성)
      - adapter_config.json             (다운로드 받은 파일)
      - adapter_model.safetensors       (다운로드 받은 파일)
      - special_tokens_map.json         (다운로드 받은 파일)
      - tokenizer.json                  (다운로드 받은 파일)
      - tokenizer_config.json           (다운로드 받은 파일)
      - training_args.bin               (다운로드 받은 파일)
    - ...
  - ...
```

### 2-2. S-BERT 모델 파일 (```qna```)

```ai_qna/models/rag_sbert``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- ai_qna
  - models                                                     (필요 시 디렉토리 새로 생성)
    - rag_sbert                                                (디렉토리 새로 생성)
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

### 2-3. ```quiz``` LLM 파일

```ai_quiz/models/kananai_sft_final_fine_tuned_10epochs``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- ai_quiz
  - models                                       (필요 시 디렉토리 새로 생성)
    - kananai_sft_final_fine_tuned_10epochs      (디렉토리 새로 생성)
      - adapter_config.json                      (다운로드 받은 파일)
      - adapter_model.safetensors                (다운로드 받은 파일)
      - special_tokens_map.json                  (다운로드 받은 파일)
      - tokenizer.json                           (다운로드 받은 파일)
      - tokenizer_config.json                    (다운로드 받은 파일)
      - training_args.bin                        (다운로드 받은 파일)
    - ...
  - ...
```

### 2-4. S-BERT 모델 파일 (```quiz```)

```ai_quiz/models/sbert``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- ai_quiz
  - models                                                     (필요 시 디렉토리 새로 생성)
    - sbert                                                    (디렉토리 새로 생성)
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

### 2-5. ```interview``` LLM 파일

```ai_interview/models/kananai_sft_final_fine_tuned_5epochs``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- ai_interview
  - models                                      (필요 시 디렉토리 새로 생성)
    - kananai_sft_final_fine_tuned_5epochs      (디렉토리 새로 생성)
      - adapter_config.json                     (다운로드 받은 파일)
      - adapter_model.safetensors               (다운로드 받은 파일)
      - special_tokens_map.json                 (다운로드 받은 파일)
      - tokenizer.json                          (다운로드 받은 파일)
      - tokenizer_config.json                   (다운로드 받은 파일)
      - training_args.bin                       (다운로드 받은 파일)
    - ...
  - ...
```

### 2-6. S-BERT 모델 파일 (```interview``` - 사용자 답변 성공 여부 파악)

```ai_interview/models/output_answer_sbert``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- ai_interview
  - models                                                     (필요 시 디렉토리 새로 생성)
    - output_answer_sbert                                      (디렉토리 새로 생성)
      - trained_sbert_model_40                                 (디렉토리 새로 생성)
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

### 2-7. S-BERT 모델 파일 (```interview``` - 다음 질문 선택)

```ai_interview/models/next_question_sbert``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- ai_interview
  - models                                                     (필요 시 디렉토리 새로 생성)
    - next_question_sbert                                      (디렉토리 새로 생성)
      - trained_sbert_model_40                                 (디렉토리 새로 생성)
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

### 2-8. S-BERT 모델 파일 (Ethics)

```final_product/models/ethics_sbert/trained_sbert_model``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_07_02_OhLoRA_ML_Tutor
- final_product
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