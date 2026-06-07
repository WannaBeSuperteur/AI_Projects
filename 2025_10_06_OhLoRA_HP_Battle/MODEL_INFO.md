
## 목차

* [1. 모델 정보](#1-모델-정보)
* [2. 모델 배치 방법](#2-모델-배치-방법)
  * [2-1. Hyper-param 최적화 모델 (HPO Model)](#2-1-hyper-param-최적화-모델-hpo-model)
  * [2-2. Auto-Encoder](#2-2-auto-encoder)
  * [2-3. Oh-LoRA LLM](#2-3-oh-lora-llm)

## 1. 모델 정보

* 아래 표에 표시된 모든 모델 **(총 3개)** 은 **Oh-LoRA 👱‍♀️ (오로라) ML Tutor** 의 원활한 동작을 위해 필요
* 본 문서에 표시된 모든 저장 경로는 **처음부터 ```/2025_10_06_OhLoRA_HP_Battle/``` 까지의 경로는 제외하고 나타낸** 경로임.
* ```models``` 및 ```llm/models/kananai_sft_final_fine_tuned``` 디렉토리가 처음 clone 받았을 때는 없으므로, **해당 디렉토리를 먼저 생성 후** 진행

| 모델                                 | 용도                                                                                | 다운로드 경로                                                                               | 모델 파일 배치 방법                                      |
|------------------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|--------------------------------------------------|
| Hyper-param 최적화 모델 **(HPO Model)** | 데이터셋 특성과 하이퍼파라미터 값이 주어졌을 때, **하이퍼파라미터 최적화 대상 CNN 모델의 성능지표 (Macro F1 Score)** 값 예측 | [HuggingFace](https://huggingface.co/daebakgazua/251006_OhLoRA_HP_Battle_hpo_models)  | [모델 파일 배치 방법](#2-1-hyper-param-최적화-모델-hpo-model) |
| Auto-Encoder                       | 각 데이터셋 별 **이미지 특성으로 사용하기 위한 hidden vector 추출** 을 위한 Auto-Encoder                  | [HuggingFace](https://huggingface.co/daebakgazua/251006_OhLoRA_HP_Battle_AutoEncoder) | [모델 파일 배치 방법](#2-2-auto-encoder)                 |
| Oh-LoRA LLM                        | Oh-LoRA 👱‍♀️ vs. 인간의 Hyper-param battle 결과에 대한 텍스트 생성                            | [HuggingFace](https://huggingface.co/daebakgazua/251006_OhLoRA_HP_Battle_LLM)         | [모델 파일 배치 방법](#2-3-oh-lora-llm)                  |

## 2. 모델 배치 방법

### 2-1. Hyper-param 최적화 모델 (HPO Model)

```hpo_training_model``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_10_06_OhLoRA_HP_Battle
- hpo_training_model
  - hpo_model_cifar_10.pt               (다운로드 받은 파일)
  - hpo_model_fashion_mnist.pt          (다운로드 받은 파일)
  - hpo_model_mnist.pt                  (다운로드 받은 파일)
```

### 2-2. Auto-Encoder

```models``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_10_06_OhLoRA_HP_Battle
- models                                (필요 시 디렉토리 새로 생성)
  - ae_decoder_cifar_10.pt              (다운로드 받은 파일)
  - ae_decoder_fashion_mnist.pt         (다운로드 받은 파일)
  - ae_decoder_mnist.pt                 (다운로드 받은 파일)  
  - ae_encoder_cifar_10.pt              (다운로드 받은 파일)
  - ae_encoder_fashion_mnist.pt         (다운로드 받은 파일)
  - ae_encoder_mnist.pt                 (다운로드 받은 파일)  
  - ae_model_cifar_10.pt                (다운로드 받은 파일)
  - ae_model_fashion_mnist.pt           (다운로드 받은 파일)
  - ae_model_mnist.pt                   (다운로드 받은 파일)  
```

### 2-3. Oh-LoRA LLM

```llm/models/kananai_sft_final_fine_tuned``` 경로에 HuggingFace 링크에서 다운로드받은 파일을 아래와 같이 배치

```
2025_10_06_OhLoRA_HP_Battle
- llm
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
