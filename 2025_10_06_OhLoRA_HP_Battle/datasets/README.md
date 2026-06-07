## 목차

* [1. 개요](#1-개요)
* [2. 데이터셋 배치](#2-데이터셋-배치)
  * [2-1. CIFAR-10](#2-1-cifar-10)
  * [2-2. MNIST](#2-2-mnist)
  * [2-3. Fashion-MNIST](#2-3-fashion-mnist)
* [3. 데이터셋 생성 방법](#3-데이터셋-생성-방법)

## 1. 개요

하이퍼파라미터 튜닝 대상 데이터셋 3개

* CIFAR-10
* MNIST
* Fashion-MNIST

## 2. 데이터셋 배치

* 다운받은 압축파일을 풀어서 그 내용물 (파일) 들을 다음과 같이 배치

### 2-1. CIFAR-10

```
datasets            (본 디렉토리)
 - cifar_10         (디렉토리 생성)
   - test
     - airplane
       - 0001.png
       - 0002.png
       - 0003.png
       ...
     ...
   - train
     - airplane
       - 0001.png
       - 0002.png
       - 0003.png
       ...
     ...   
```

### 2-2. MNIST

```
datasets              (본 디렉토리)
 - mnist              (디렉토리 생성)
   - mnist_test.csv   (around 17.4 MB)
   - mnist_train.csv  (around 104 MB)
```

### 2-3. Fashion-MNIST

```
datasets                      (본 디렉토리)
 - fashion_mnist              (디렉토리 생성)
   - fashion-mnist_test.csv   (around 21.1 MB)
   - fashion-mnist_train.csv  (around 126 MB)
```

## 3. 데이터셋 생성 방법

| 데이터셋                                                                                      | 설명                                            | 이미지 해상도<br>(Channel 개수 포함) | Class 개수 | train data | test data |
|-------------------------------------------------------------------------------------------|-----------------------------------------------|----------------------------|----------|------------|-----------|
| MNIST [(download)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)             | 딥러닝 초보자의 연습용 & 모델 설명 예시용으로 널리 사용되는 숫자 분류 데이터셋 | 1 x 28 x 28                | 10       | 60,000     | 10,000    |
| Fashion MNIST [(download)](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | MNIST와 유사하되, 숫자가 아닌 옷 이미지를 사용                 | 1 x 28 x 28                | 10       | 60,000     | 10,000    |
| CIFAR-10 [(download)](https://www.kaggle.com/datasets/ayush1220/cifar10)                  | Object 분류 데이터셋                                | 3 x 32 x 32                | 10       | 50,000     | 10,000    |

* 압축을 푼 후, [데이터셋 배치](#2-데이터셋-배치) 에 맞게 데이터셋 배치
* ```python create_mnist_images.py``` 파일을 실행하여 MNIST, Fashion-MNIST 데이터셋 이미지 생성
