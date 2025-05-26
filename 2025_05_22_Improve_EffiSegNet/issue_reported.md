## 목차

* [1. 전체 이슈 요약](#1-전체-이슈-요약)
* [2. 이슈 상세](#2-이슈-상세)
  * [2-1. Augmentation 적용 시 image 에만 적용되고, mask 에는 적용 안됨 (해결 완료)](#2-1-augmentation-적용-시-image-에만-적용되고-mask-에는-적용-안됨-해결-완료)

## 1. 전체 이슈 요약

| 이슈 분류        | 이슈                                             | 날짜         | 심각성    | 상태    | 원인 (및 해결 방법)                                               | 시도했으나 실패한 해결 방법 |
|--------------|------------------------------------------------|------------|--------|-------|------------------------------------------------------------|-----------------|
| Augmentation | Augmentation 적용 시 image 에만 적용되고, mask 에는 적용 안됨 | 2025.05.24 | **보통** | 해결 완료 | ```ImageOnlyTransform``` 적용<br>→ ```DualTransform``` 으로 수정 | -               |

## 2. 이슈 상세

### 2-1. Augmentation 적용 시 image 에만 적용되고, mask 에는 적용 안됨 (해결 완료)

**1. 문제 상황**

* [왼쪽 위에 직사각형을 추가하는 Augmentation](effisegnet_improved/README.md#2-5--black-rectangle-추가) 적용 시,
  * 추가된 직사각형 부분을 mask 에도 ```종양이 아님``` 영역으로 지정하여 반영해야 함
  * 그러나, mask 에는 반영이 안 되고 오직 image 에만 반영됨

**2. 문제 원인 및 해결 방법**

* 문제 원인
  * 해당 Augmentation 을 나타내는 Class 인 ```AddBlackRectangleAtTopLeft``` 의 super class 로 ```ImageOnlyTransform``` 을 적용함
  * ```ImageOnlyTransform``` 은 **image 에만 적용되고 mask 에는 적용이 안 되는** Augmentation 을 나타냄
* 해결 방법
  * 아래와 같이 **image, mask 에 둘 다 적용** 되는 ```DualTransform``` 을 적용

```python
class AddBlackRectangleAtTopLeft(DualTransform):  # ImageOnlyTransform 이면 image 에만 적용되고 mask 에는 적용 안됨
    def __init__(self, fill_value, p=0.5):
        super(AddBlackRectangleAtTopLeft, self).__init__(always_apply=False, p=p)
        self.fill_value = fill_value

    def apply(self, img, **params):
        height, width = img.shape[:2]

        ...
```

**3. 교훈**

* Augmentation class 를 정의할 때, super class 가 무엇을 의미하는지 파악하자.
* 더 나아가, 코드를 작성하거나 기존 코드를 갖다 재사용할 때 **그 의미를 파악하는 습관을 더욱 강하게 기르자.**
