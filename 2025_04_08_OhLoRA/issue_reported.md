## 목차

* [1. 전체 이슈 요약](#1-전체-이슈-요약)
* [2. 이슈 상세](#2-이슈-상세)
  * [2-1. StyleGAN Fine-Tuning Tensor 자료형 불일치](#2-1-stylegan-fine-tuning-tensor-자료형-불일치) 
  * [2-2. StyleGAN-FineTune-v2 이미지 생성 테스트 시 Memory Leak](#2-2-stylegan-finetune-v2-이미지-생성-테스트-시-memory-leak)
  * [2-3. LLM Fine-Tuning 시 Batch size 오류](#2-3-llm-fine-tuning-시-batch-size-오류)

## 1. 전체 이슈 요약

| 이슈 분류    | 이슈                                            | 날짜         | 심각성    | 상태    | 원인 (및 해결 방법)                                                                                  | 시도했으나 실패한 해결 방법                                                                        |
|----------|-----------------------------------------------|------------|--------|-------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| StyleGAN | Fine-tuning 시 Tensor 자료형 불일치                  | 2025.04.12 | **심각** | 해결 완료 | property score vector 를 Float32 로 type casting 하여 해결                                          | - property score vector 를 Float64 로 type casting **(해결 안됨)**                           |
| StyleGAN | StyleGAN-FineTune-v2 이미지 생성 테스트 시 Memory Leak | 2025.04.15 | 보통     | 해결 완료 | ```with torch.no_grad()``` 없이 CUDA 에 올라온 tensor 를 이용하여 이미지 생성                                 |                                                                                        |
| LLM      | LLM Fine-Tuning 시 Batch size 오류               | 2025.04.21 | **심각** | 해결 완료 | 인식 가능한 CUDA GPU 가 2대인 상황에서 **실제 batch size = (GPU 개수 * 의도한 batch size)** 를 적용, batch size 불일치 | - 인식 가능한 CUDA GPU 제한 (잘못된 코드 위치)<br>- LLM Training Config 에서 ```use_liger = True``` 설정 | 

## 2. 이슈 상세

### 2-1. StyleGAN Fine-Tuning Tensor 자료형 불일치

**1. 문제 상황 및 원인 요약**

* StyleGAN 에서 인물 특징을 나타내는 property score ([참고](stylegan_and_segmentation/README.md#2-핵심-속성-값)) 벡터인 ```concatenated_labels``` 의 Tensor 를 StyleGAN Fine-Tuning 을 위해 StyleGAN 에 입력 시,
* StyleGAN 에서 **해당 Tensor 와 Label Weight 행렬을 matmul 할 때 자료형 불일치 (double != float) 오류 발생**

```
  File "C:\Users\20151\Documents\AI_Projects\2025_04_08_OhLoRA\stylegan_and_segmentation\stylegan_modified\stylegan_generator.py", line 287, in forward
    embedding = torch.matmul(label, self.label_weight)
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: double != float
```

* 출력 결과

| Tensor                                         | 출력 결과                                                                                       |
|------------------------------------------------|---------------------------------------------------------------------------------------------|
| label (property score 를 나타낸 벡터)                | ```[-1.7069, -0.7809,  0.5124, -0.4403,  0.0337]], device='cuda:0', dtype=torch.float64)``` |
| self.label_weight (StyleGAN 의 Label Weight 행렬) | ```[ 2.8673,  1.6306,  0.5414,  ...,  0.3971, -0.9557, -1.0875]], device='cuda:0')```       |

**2. 해결 시도 방법**

* 1. ```concatenated_labels``` 를 **Double** 로 Type Casting
  * 실패 (이미 Float64 = Double 임)
* 2. ```concatenated_labels``` 를 **Float32** 로 Type Casting
  * 성공 🎉

**3. 교훈**

* Float64 = Double 임을 확실히 알아 두자.

### 2-2. StyleGAN-FineTune-v2 이미지 생성 테스트 시 Memory Leak

**1. 문제 상황 및 원인 요약**

* [StyleGAN-FineTune-v2](stylegan_and_segmentation/README.md#3-1-image-generation-model-stylegan) 학습 중 이미지 생성 테스트 시, Memory Leak 발생
* 이미지 생성 코드에서, ```with torch.no_grad()``` 없이 CUDA 에 올라온 tensor 를 이용하여 이미지를 생성했기 때문에 Memory Leak 발생

**2. 해결 시도 방법**

* 이미지 생성 코드를 ```with torch.no_grad()``` 로 감쌈
  * [해당 commit](https://github.com/WannaBeSuperteur/AI_Projects/commit/d063afb17016a1b08b15b68102e679b8c302d109) 
  * 결과 : 해결 성공 🎉 

**3. 교훈**

* 모델의 Train 없이 Validation 또는 Inference 만 필요한 작업에서는 ```with torch.no_grad()``` 를 **반드시** 사용하자.

### 2-3. LLM Fine-Tuning 시 Batch size 오류

**1. 문제 상황 및 원인 요약**

```
Traceback (most recent call last):
  File "C:\Users\20151\Documents\AI_Projects\2025_04_08_OhLoRA\llm\unsloth_test\test_without_unsloth.py", line 78, in <module>
    trainer.train()
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\transformers\trainer.py", line 2245, in train
    return inner_training_loop(
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\transformers\trainer.py", line 2560, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\transformers\trainer.py", line 3736, in training_step
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\trl\trainer\sft_trainer.py", line 490, in compute_loss
    correct_predictions = (predictions == shift_labels) & mask
RuntimeError: The size of tensor a (2) must match the size of tensor b (4) at non-singleton dimension 0
```

* [Unsloth 사용 시 속도 및 메모리 사용량 테스트](llm/README.md#4-2-unsloth-use-test) 중, **Train batch size 와 실제 prediction 결과의 batch size 가 맞지 않아서 size error 발생**
* 인식 가능한 CUDA GPU ```torch.cuda.device_count()``` 는 2대인데, LLM 의 Fine-Tuning 에 필요한 Trainer 객체는 **각 CUDA GPU 별로 Batch Size 를 계산** 하는 방식
  * 이로 인해 인식 가능한 CUDA GPU가 2대이면, **의도한 batch size 가 2 일 때 (실제 train batch size) = 2 * 2 = 4** 가 되기 때문에 크기가 맞지 않는 오류 발생

**2. 해결 시도 방법**

* ```os.environ["CUDA_VISIBLE_DEVICES"] = "0"``` 을 ```test_without_unsloth.py``` 의 상단에 추가
  * **방법은 맞지만, 코드 위치가 잘못됨**  
  * 인식 가능한 GPU 는 여전히 2대
  * 따라서 여전히 train batch size = 4 이며, 오류 발생
* ```use_liger=True``` 강제 설정
  * 이렇게 하면 trl 라이브러리 코드 구조상 해당 오류를 회피 가능
  * ❌ 근본적인 해결 방법이 아님
  * ❌ 아래와 같이 **CUDA OOM** 발생

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 12.00 GiB of which 0 bytes is free. Of the allocated memory 10.90 GiB is allocated by PyTorch, and 518.10 MiB is reserved by 
PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

* ```os.environ["CUDA_VISIBLE_DEVICES"] = "0"``` 을 ```test_unsloth_common.py``` 의 상단에 추가
  * 🎉 이 방법으로 **해결 성공**
  * 인식 가능한 GPU 를 **1대로 제한하는 올바른 방법이자 근본적인 해결책**
  * **방법과 코드 위치가 모두 알맞음**
    * train batch size 가 4 가 되는 원인은 training argument (SFTConfig) 에 있음
    * training argument 는 ```test_without_unsloth.py``` 의 최상단에서 ```from test_unsloth_common import ...``` 할 때 이미 정의가 완료되므로, 위 **잘못된 위치** 의 경우는 인식 가능한 GPU 대수 제한 설정이 **이미 늦은 것임**
  * 참고
    * 실제 transformers==4.51.3 라이브러리에서의 ```train_batch_size``` 값 결정 코드
    * ```training_args.py``` Line 2286
      * ```self._n_gpu = torch.cuda.device_count()```
    * ```training_args.py``` Line 2137
      * ```train_batch_size = per_device_batch_size * max(1, self.n_gpu)```