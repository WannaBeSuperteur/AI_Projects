
* EffiSegNet Improved Implementation
  * EffiSegNet Base Implementation is from [Official PyTorch Implementation](https://github.com/ivezakis/effisegnet/tree/main)

## 목차

* [1. TEST RESULT](#1-test-result)
* [2. 변경점 상세](#2-변경점-상세)
  * [2-1. 📈 ColorJitter & Affine Prob 상향](#2-1--colorjitter--affine-prob-상향) 
  * [2-2. ✂ Remove ElasticTransform](#2-2--remove-elastictransform)
  * [2-3. 🌬 Weaken ColorJitter](#2-3--weaken-colorjitter)
  * [2-4. 📐 Inference Prediction Threshold 조정](#2-4--inference-prediction-threshold-조정)
  * [2-5. ⬛ Black Rectangle 추가](#2-5--black-rectangle-추가)
  * [2-6. 🧭 Near-Pixel-Diff Loss Term 추가](#2-6--near-pixel-diff-loss-term-추가)

## 1. TEST RESULT

* with **EffiSegNet-B4 (Pre-trained) & 50 epochs** option (instead of 300 epochs option of original paper)
* [Test Result Details](https://github.com/WannaBeSuperteur/AI_Projects/issues/13)

![image](../../images/250522_2.png)

* 변경점 이모지

| 이모지 | 변경점                                                       |
|-----|-----------------------------------------------------------|
| 📈  | ColorJitter Prob. 0.5 → 0.8<br>Affine Prob. 0.5 → 0.8     |
| ✂   | ElasticTransform Augmentation 제거                          |
| 🌬  | ColorJitter Augmentation 강도 약화                            |
| ⬛   | 이미지의 좌측 상단에 검은색 직사각형 추가                                   |
| 📐  | Inference 시, Prediction Threshold 0.5 → sigmoid(0.5) 로 조정 |
| 🧭  | Near-Pixel-Diff Loss Term 추가                              |

* 각 모델 별 변경점 및 성능

| 모델                  | 변경점   | Test Dice             | Test IoU              | Test Recall           | Test Precision        |
|---------------------|-------|-----------------------|-----------------------|-----------------------|-----------------------|
| Original EffiSegNet |       | 0.9310                | 0.8803                | 0.9363                | 0.9538                |
| 1차 수정 (05.24)       | 📈✂   | 0.9406 (▲ 0.0096)     | 0.8913 (▲ 0.0110)     | 0.9259 (▼ 0.0104)     | **0.9626 (▲ 0.0088)** |
| 2차 수정 (05.24)       | 📈    | 0.9363 (▲ 0.0053)     | 0.8860 (▲ 0.0057)     | 0.9295 (▼ 0.0068)     | 0.9587 (▲ 0.0049)     |
| 3차 수정 (05.24)       | ✂     | 0.9386 (▲ 0.0076)     | 0.8904 (▲ 0.0101)     | 0.9389 (▲ 0.0026)     | 0.9559 (▲ 0.0021)     |
| 4차 수정 (05.24)       | 📈✂🌬 | **0.9421 (▲ 0.0111)** | **0.8944 (▲ 0.0141)** | 0.9385 (▲ 0.0022)     | 0.9590 (▲ 0.0052)     |
| 5차 수정 (05.24)       | ⬛     | 0.9370 (▲ 0.0060)     | 0.8879 (▲ 0.0076)     | 0.9378 (▲ 0.0015)     | 0.9568 (▲ 0.0030)     |
| 6차 수정 (05.25)       | 📐    | 0.9304 (▼ 0.0006)     | 0.8790 (▼ 0.0013)     | 0.9295 (▼ 0.0068)     | 0.9594 (▲ 0.0056)     |
| 7차 수정 (05.25)       | 🧭    | 0.9347 (▲ 0.0037)     | 0.8824 (▲ 0.0021)     | **0.9528 (▲ 0.0165)** | 0.9409 (▼ 0.0129)     |
| 8차 수정 (05.25)       | ⬛🧭   | 0.9315 (▲ 0.0005)     | 0.8772 (▼ 0.0031)     | 0.9501 (▲ 0.0138)     | 0.9366 (▼ 0.0172)     |

## 2. 변경점 상세

### 2-1. 📈 ColorJitter & Affine Prob 상향

### 2-2. ✂ Remove ElasticTransform

### 2-3. 🌬 Weaken ColorJitter

### 2-4. 📐 Inference Prediction Threshold 조정

### 2-5. ⬛ Black Rectangle 추가

### 2-6. 🧭 Near-Pixel-Diff Loss Term 추가