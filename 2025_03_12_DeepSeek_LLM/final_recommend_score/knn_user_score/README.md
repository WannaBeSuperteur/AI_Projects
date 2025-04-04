## 목차

* [1. 개요](#1-개요)
* [2. 파일 저장 방법](#2-파일-저장-방법)

## 1. 개요

LLM answer 를 통해 생성되는 각 다이어그램에 대한 최종 평가 점수에 반영되는 **예상 사용자 평가 점수** 를 [k-NN (K-Nearest Neighbors)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%AA%A8%EB%8D%B8_KNN.md) 알고리즘으로 계산하기 위한 학습 데이터 저장 공간입니다.

![image](../../../images/250312_16.PNG)

**알고리즘 동작 방식 (위 그림 참고)**

* 기존 사용자 평가 데이터 (이 디렉토리의 하위 디렉토리들에 저장된 Diagram 원본 이미지) 를 **일정한 방법으로 변환** 후, Auto-Encoder 로 저차원 벡터로 변환합니다.
* 새로 입력되는 Diagram 원본 이미지를 **기존 사용자 평가 데이터와 동일한 방법으로 변환** 후, 마찬가지로 Auto-Encoder 로 저차원 벡터로 변환합니다.
* 새로 입력된 Diagram 과 가장 가까운 기존 사용자 평가 데이터의 Diagram 들을 K 개 찾습니다.
* **K 개의 Diagram 에 대해 사용자가 매긴 점수 (디렉토리 이름) 의 가중 평균** 을 **예상 사용자 평가 점수 (위 그림의 Final Predicted Score)** 로 계산합니다.
  * 가중치는 거리가 클수록 감소합니다. 

## 2. 파일 저장 방법

* 본 디렉토리에 ```5``` ```4``` ```3``` ```2``` ```1``` ```0``` 이라는 이름의 각 하위 디렉토리를 생성합니다.
* 해당 하위 디렉토리 이름에 해당하는 점수를 매기고 싶은 Diagram 원본 이미지를 해당 폴더에 넣습니다.
* 상위 디렉토리인 ```final_recommend_score``` 에 있는 ```knn.py``` 를 실행하면, ```final_recommend_score/diagrams_for_test``` 디렉토리에 있는 새로 입력되는 다이어그램인 ```test_diagram_{i}.png``` (```i``` = 0, 1, 2, ...) 에 대한 **예상 사용자 평가 점수** 를 계산하여 출력합니다. 