## 목차

* [1. 모델 정보](#1-모델-정보)
  * [1-1. 기존 Pre-trained 모델](#1-1-기존-pre-trained-모델)
  * [1-2. Oh-LoRA 프로젝트 용 모델](#1-2-oh-lora-프로젝트-용-모델)
* [2. 데이터셋 정보](#2-데이터셋-정보)

## 1. 모델 정보

### 1-1. 기존 Pre-trained 모델

| 모델 분류        | 모델 파일 이름<br>(원래 이름에서 아래 이름으로 변경)                                                                                                              | 저장 위치 (디렉토리)<br>(```2025_04_08_OhLoRA``` 까지의 경로 제외) | 다운로드 주소 (출처)                                                                                                              |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| StyleGAN     | ```stylegan_model.pth```                                                                                                                      | ```stylegan_and_segmentation/stylegan```            | [GenForce GitHub](https://github.com/genforce/genforce/blob/master/MODEL_ZOO.md) > StyleGAN Ours > celeba_partial-256x256 |
| Segmentation | ```mtcnn_onet.pt```                                                                                                                           | ```stylegan_and_segmentation/segmentation/models``` | [FaceNet Timesler GitHub](https://github.com/timesler/facenet-pytorch/blob/master/data)                                   |
| Segmentation | ```mtcnn_pnet.pt```                                                                                                                           | ```stylegan_and_segmentation/segmentation/models``` | [FaceNet Timesler GitHub](https://github.com/timesler/facenet-pytorch/blob/master/data)                                   |
| Segmentation | ```mtcnn_rnet.pt```                                                                                                                           | ```stylegan_and_segmentation/segmentation/models``` | [FaceNet Timesler GitHub](https://github.com/timesler/facenet-pytorch/blob/master/data)                                   |
| Segmentation | ```segmentation_model.pt```                                                                                                                   | ```stylegan_and_segmentation/segmentation/models``` | [FaceXFormer HuggingFace](https://huggingface.co/kartiknarayan/facexformer/tree/main/ckpts)                               |
| LLM          | 파일 배치 방법 :<br>- [해당 문단](llm/README.md#4-1-prepare-model-gemma-2-2b-based) > **"1. Gemma-2-2b Original Unsloth Model (by Google & Unsloth)"**  | ```llm/models/original```                           | [Gemma-2 2B HuggingFace](https://huggingface.co/unsloth/gemma-2-2b-it/tree/main)                                          |
| LLM          | 파일 배치 방법 :<br>- [해당 문단](llm/README.md#4-2-prepare-model-polyglot-ko-13b-based) > **"1. Polyglot-Ko Original Model (by EleutherAI, ✅ 최종 채택)"** | ```llm/models/polyglot_original```                  | [Polyglot-Ko 1.3B HuggingFace](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main)                              |
| LLM S-BERT   | 파일 배치 방법 :<br>- [해당 문단](llm/README.md#4-3-prepare-s-bert-model) > **"1. Final Fine-Tuned S-BERT Model"**                                      | ```llm/models/memory_sbert/trained_sbert_model```   | [RoBERTa-base HuggingFace](https://huggingface.co/klue/roberta-base/tree/main)                                            |                                      

### 1-2. Oh-LoRA 프로젝트 용 모델

* 모델 이름 끝의 ✅ 표시는 **실제 Oh-LoRA 👱‍♀️ (오로라) 사용을 위해 필요** 한 모델을 나타냄

| 모델 분류      | 모델 이름                                 | 모델 파일 이름<br>(원래 이름에서 아래 이름으로 변경) | 저장 위치 (디렉토리)<br>(```2025_04_08_OhLoRA``` 까지의 경로 제외) | 다운로드 주소 (출처) |
|------------|---------------------------------------|----------------------------------|-----------------------------------------------------|--------------|
| CNN        | CNN (Gender Score)                    |                                  |                                                     |              |
| CNN        | CNN (Quality Score)                   |                                  |                                                     |              |
| CNN        | CNN (7 Property Scores)               |                                  |                                                     |              |
| StyleGAN   | StyleGAN-FineTuned-v1 (Generator)     |                                  |                                                     |              |
| StyleGAN   | StyleGAN-FineTuned-v1 (Discriminator) |                                  |                                                     |              |
| StyleGAN   | StyleGAN-FineTuned-v3 (Generator) ✅   |                                  |                                                     |              |
| StyleGAN   | StyleGAN-FineTuned-v4 (Generator)     |                                  |                                                     |              |
| StyleGAN   | StyleGAN-FineTuned-v4 (Discriminator) |                                  |                                                     |              |
| LLM        | OhLoRA LLM ✅                          |                                  |                                                     |              |
| LLM S-BERT | S-BERT model for OhLoRA LLM memory ✅  |                                  |                                                     |              |

## 2. 데이터셋 정보

| 데이터셋 이름                                    | 데이터셋 파일 이름 | 저장 위치 (디렉토리)<br>(```2025_04_08_OhLoRA``` 까지의 경로 제외) | 다운로드 주소 (출처) |
|--------------------------------------------|------------|-----------------------------------------------------|--------------|
| StyleGAN Generated 10K images              |            |                                                     |              |
| StyleGAN Generated 4,703 images (Filtered) |            |                                                     |              |
