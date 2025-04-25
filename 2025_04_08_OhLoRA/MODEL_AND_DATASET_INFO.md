## 목차

* [1. 모델 정보](#1-모델-정보)
  * [1-1. 기존 Pre-trained 모델](#1-1-기존-pre-trained-모델)
  * [1-2. Oh-LoRA 프로젝트 용 모델](#1-2-oh-lora-프로젝트-용-모델)
* [2. 데이터셋 정보](#2-데이터셋-정보)
* [3. 실제 Oh-LoRA 👱‍♀️ (오로라) 사용을 위해 필요한 LLM 및 S-BERT 모델 배치 방법](#3-실제-oh-lora--오로라-사용을-위해-필요한-llm-및-s-bert-모델-배치-방법)
  * [3-1. OhLoRA LLM](#3-1-ohlora-llm)
  * [3-2. S-BERT model for OhLoRA LLM memory](#3-2-s-bert-model-for-ohlora-llm-memory)

## 1. 모델 정보

### 1-1. 기존 Pre-trained 모델

| 모델 분류        | 모델 파일 이름<br>(🔄 : 원래 이름에서 아래 이름으로 변경)                                                                                                         | 저장 위치 (디렉토리)<br>(```2025_04_08_OhLoRA``` 까지의 경로 제외) | 다운로드 주소 (출처)                                                                                                              |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| StyleGAN     | ```stylegan_model.pth``` 🔄                                                                                                                   | ```stylegan_and_segmentation/stylegan```            | [GenForce GitHub](https://github.com/genforce/genforce/blob/master/MODEL_ZOO.md) > StyleGAN Ours > celeba_partial-256x256 |
| Segmentation | ```mtcnn_onet.pt```                                                                                                                           | ```stylegan_and_segmentation/segmentation/models``` | [FaceNet Timesler GitHub](https://github.com/timesler/facenet-pytorch/blob/master/data)                                   |
| Segmentation | ```mtcnn_pnet.pt```                                                                                                                           | ```stylegan_and_segmentation/segmentation/models``` | [FaceNet Timesler GitHub](https://github.com/timesler/facenet-pytorch/blob/master/data)                                   |
| Segmentation | ```mtcnn_rnet.pt```                                                                                                                           | ```stylegan_and_segmentation/segmentation/models``` | [FaceNet Timesler GitHub](https://github.com/timesler/facenet-pytorch/blob/master/data)                                   |
| Segmentation | ```segmentation_model.pt``` 🔄                                                                                                                | ```stylegan_and_segmentation/segmentation/models``` | [FaceXFormer HuggingFace](https://huggingface.co/kartiknarayan/facexformer/tree/main/ckpts)                               |
| LLM          | 파일 배치 방법 :<br>- [해당 문단](llm/README.md#4-1-prepare-model-gemma-2-2b-based) > **"1. Gemma-2-2b Original Unsloth Model (by Google & Unsloth)"**  | ```llm/models/original```                           | [Gemma-2 2B HuggingFace](https://huggingface.co/unsloth/gemma-2-2b-it/tree/main)                                          |
| LLM          | 파일 배치 방법 :<br>- [해당 문단](llm/README.md#4-2-prepare-model-polyglot-ko-13b-based) > **"1. Polyglot-Ko Original Model (by EleutherAI, ✅ 최종 채택)"** | ```llm/models/polyglot_original```                  | [Polyglot-Ko 1.3B HuggingFace](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main)                              |
| LLM S-BERT   | 파일 배치 방법 :<br>- [해당 문단](llm/README.md#4-3-prepare-s-bert-model) > **"1. Final Fine-Tuned S-BERT Model"**                                      | ```llm/models/memory_sbert/trained_sbert_model```   | [RoBERTa-base HuggingFace](https://huggingface.co/klue/roberta-base/tree/main)                                            |                                      

### 1-2. Oh-LoRA 프로젝트 용 모델

* 모델 이름 끝의 ✅ 표시는 **실제 Oh-LoRA 👱‍♀️ (오로라) 사용을 위해 필요** 한 모델을 나타냄

| 모델 분류      | 모델 이름<br>(상세 정보 링크)                                                                                              | 모델 파일 이름<br>(모두 원래 이름 그대로)                                     | 저장 위치 (디렉토리)<br>(```2025_04_08_OhLoRA``` 까지의 경로 제외) | 다운로드 주소 (출처)                                                                                  |
|------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------|
| CNN        | [CNN (Gender Score)](stylegan_and_segmentation/README.md#3-2-cnn-model-성별-이미지-품질)                                | ```gender_model_{k}.pt```, k = 0,1,2,3,4                       | ```stylegan_and_segmentation/cnn/models```          | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_CNNs/tree/main)               |
| CNN        | [CNN (Quality Score)](stylegan_and_segmentation/README.md#3-2-cnn-model-성별-이미지-품질)                               | ```quality_model_{k}.pt```, k = 0,1,2,3,4                      | ```stylegan_and_segmentation/cnn/models```          | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_CNNs/tree/main)               |
| CNN        | [CNN (7 Property Scores)](stylegan_and_segmentation/README.md#3-3-cnn-model-나머지-핵심-속성-값-7개)                      | ```stylegan_gen_fine_tuned_v2_cnn.pth```                       | ```stylegan_and_segmentation/stylegan_modified```   | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_StyleGAN_FineTuned/tree/main) |
| StyleGAN   | [StyleGAN-FineTuned-v1 (Generator)](stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan)     | ```stylegan_gen_fine_tuned_v1.pth```                           | ```stylegan_and_segmentation/stylegan_modified```   | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_StyleGAN_FineTuned/tree/main) |
| StyleGAN   | [StyleGAN-FineTuned-v1 (Discriminator)](stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan) | ```stylegan_dis_fine_tuned_v1.pth```                           | ```stylegan_and_segmentation/stylegan_modified```   | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_StyleGAN_FineTuned/tree/main) |
| StyleGAN   | [StyleGAN-FineTuned-v3 (Generator) ✅](stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan)   | ```stylegan_gen_fine_tuned_v3_ckpt_0005_gen.pth```             | ```stylegan_and_segmentation/stylegan_modified```   | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_StyleGAN_FineTuned/tree/main) |
| StyleGAN   | [StyleGAN-FineTuned-v4 (Generator)](stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan)     | ```stylegan_gen_fine_tuned_v4.pth```                           | ```stylegan_and_segmentation/stylegan_modified```   | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_StyleGAN_FineTuned/tree/main) |
| StyleGAN   | [StyleGAN-FineTuned-v4 (Discriminator)](stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan) | ```stylegan_dis_fine_tuned_v4.pth```                           | ```stylegan_and_segmentation/stylegan_modified```   | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_StyleGAN_FineTuned/tree/main) |
| LLM        | [OhLoRA LLM ✅](llm/README.md#2-how-to-run-fine-tuning)                                                           | 파일 배치 방법 : [해당 문단](#3-1-ohlora-llm) 참고                         | ```llm/models/polyglot_fine_tuned```                | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_LLM/tree/main)                |
| LLM S-BERT | [S-BERT model for OhLoRA LLM memory ✅](llm/README.md#3-llm-memory-rag-like-concept)                              | 파일 배치 방법 : [해당 문단](#3-2-s-bert-model-for-ohlora-llm-memory) 참고 | ```llm/models/memory_sbert/trained_sbert_model```   | [Hugging Face](https://huggingface.co/daebakgazua/250408_OhLoRA_LLM_SBERT/tree/main)          |

## 2. 데이터셋 정보

| 데이터셋 이름                                    | 데이터셋 파일 이름<br>(모두 원래 이름 그대로)                                         | 저장 위치 (디렉토리)<br>(```2025_04_08_OhLoRA``` 까지의 경로 제외)                  | 다운로드 주소 (출처)                                                                                   |
|--------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| StyleGAN Generated 10K images              | ```000000.jpg``` ```000001.jpg``` ... ```009999.jpg``` (10K files)   | ```stylegan_and_segmentation/stylegan/synthesize_results```          | [Hugging Face](https://huggingface.co/datasets/daebakgazua/250408_OhLoRA_all_generated_images) |
| StyleGAN Generated 4,703 images (Filtered) | ```000004.jpg``` ```000005.jpg``` ... ```009999.jpg``` (4,703 files) | ```stylegan_and_segmentation/stylegan/synthesize_results_filtered``` | [Hugging Face](https://huggingface.co/datasets/daebakgazua/250408_OhLoRA_filtered_images)      |

## 3. 실제 Oh-LoRA 👱‍♀️ (오로라) 사용을 위해 필요한 LLM 및 S-BERT 모델 배치 방법

* [참고 : LLM README.md 의 모델 실행 방법 설명](llm/README.md#4-test--run-model)

### 3-1. OhLoRA LLM

* ```2025_04_08_OhLoRA/llm/models/polyglot_fine_tuned``` 에, [HuggingFace Link](https://huggingface.co/daebakgazua/250408_OhLoRA_LLM/tree/main) 로부터 다운로드 받은 모델 저장
  * 총 8 개 파일 (각종 정보 포함)
  * 이때 ```models/polyglot_fine_tuned``` 디렉토리는 Clone 받은 repo. 에 원래 없으므로, 새로 생성

```
2025_04_08_OhLoRA
- llm
  - fine_tuning
  - model_selection
  - models                                 (필요 시 디렉토리 새로 생성)
    - polyglot_fine_tuned                  (디렉토리 새로 생성)
      - .gitarrtibutes                     (다운로드 받은 파일)
      - adapter_config.json                (다운로드 받은 파일)
      - adapter_model.safetensors          (다운로드 받은 파일)
      - README.md                          (다운로드 받은 파일)
      - special_tokens_map.json            (다운로드 받은 파일)
      - tokenizer.json                     (다운로드 받은 파일)
      - tokenizer_config.json              (다운로드 받은 파일)
      - training_args.bin                  (다운로드 받은 파일)
    - ...
  - unsloth_test
  - ...  
```

### 3-2. S-BERT model for OhLoRA LLM memory

* ```2025_04_08_OhLoRA/llm/models/memory_sbert/trained_sbert_model``` 에, [HuggingFace Link](https://huggingface.co/daebakgazua/250408_OhLoRA_LLM_SBERT/tree/main) 로부터 다운로드 받은 모델 저장
  * 총 12 개 파일 (각종 정보 포함)
  * 이때 ```models/memory_sbert/trained_sbert_model``` 디렉토리는 Clone 받은 repo. 에 원래 없으므로, 새로 생성

```
2025_04_08_OhLoRA
- llm
  - fine_tuning
  - model_selection
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
        - README.md                                            (다운로드 받은 파일)
        - sentence_bert_config.json                            (다운로드 받은 파일)
        - special_tokens_map.json                              (다운로드 받은 파일)
        - tokenizer.json                                       (다운로드 받은 파일)
        - tokenizer_config.json                                (다운로드 받은 파일)
        - vocab.txt                                            (다운로드 받은 파일)
    - ...
  - unsloth_test
  - ...  
```