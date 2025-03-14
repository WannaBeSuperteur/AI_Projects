## 목차

## 1. 프로젝트 개요

* DeepSeek LLM 을 이용하여 머신러닝 프로세스 또는 딥러닝 모델 등을 설명하기 위한 **Diagram 을 생성하기 위한 정형화된 포맷의 텍스트** 를 생성한다.
* 해당 텍스트를 이용하여 일반 알고리즘으로 Diagram 을 생성한다.
* 다음과 같은 방법을 이용하여 보다 가독성 좋은 Diagram 을 생성한다.
  * **[DPO 또는 ORPO](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_기초_Fine_Tuning_DPO_ORPO.md) 와 같은 기술로 LLM 을 Fine-Tuning** 하여, LLM 자체적으로 사용자 입장에서 가독성 높은 Diagram 생성
* 가독성이 더욱 향상된 Diagram 생성을 위해 다음을 적용한다.
  * 여러 개의 Diagram 을 생성한 후, **기본 가독성 점수 + 예상 사용자 평가 점수** 가 높은 순으로 정렬하여 상위권의 Diagram 들을 사용자에게 표시
  * [CNN (Conv. Neural Network)](https://github.com/WannaBeSuperteur/AI-study/blob/main/Image%20Processing/Basics_CNN.md) 을 이용하여 가독성 높은 Diagram 인지의 **기본 가독성 점수** 산출
  * 생성된 이미지를 [Auto-Encoder](https://github.com/WannaBeSuperteur/AI-study/blob/main/Generative%20AI/Basics_Auto%20Encoder.md) 로 저차원 벡터화하고, [k-Nearest Neighbor] 의 아이디어를 이용하여 **예상 사용자 평가 점수** 산출

### 1-1. 프로젝트 진행 배경

## 2. 기술 분야 및 사용 기술

### 2-1. 관련 논문

### 2-2. 사용한 Python 라이브러리

## 3. 프로젝트 일정

## 4. 프로젝트 상세 설명

### 4-1. 도식 생성을 위한 LLM 프롬프트

### 4-2. LLM Fine-Tuning

### 4-3. 생성된 이미지의 저차원 벡터화

### 4-4. 최종 이미지 생성 및 순위 산출
