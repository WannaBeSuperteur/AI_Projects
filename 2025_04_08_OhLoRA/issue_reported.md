## 목차

* [1. 전체 이슈 요약](#1-전체-이슈-요약)
* [2. 이슈 상세](#2-이슈-상세)
  * [2-1. StyleGAN Fine-Tuning Tensor 자료형 불일치](#2-1-stylegan-fine-tuning-tensor-자료형-불일치) 

## 1. 전체 이슈 요약

| 이슈 분류    | 이슈                           | 날짜         | 심각성    | 상태    | 원인 (및 해결 방법)                                         | 시도했으나 실패한 해결 방법                                              |
|----------|------------------------------|------------|--------|-------|------------------------------------------------------|--------------------------------------------------------------|
| StyleGAN | Fine-tuning 시 Tensor 자료형 불일치 | 2025.04.12 | **심각** | 해결 완료 | property score vector 를 Float32 로 type casting 하여 해결 | - property score vector 를 Float64 로 type casting **(해결 안됨)** |

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
  * 성공

**3. 교훈**

* Float64 = Double 임을 확실히 알아 두자.