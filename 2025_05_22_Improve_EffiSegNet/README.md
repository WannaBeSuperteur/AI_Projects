## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. 성능 향상 방법](#1-1-성능-향상-방법)
  * [1-2. 성능 향상 결과](#1-2-성능-향상-결과)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 관련 논문](#2-1-관련-논문)
  * [2-2. 사용한 Python 라이브러리 및 시스템 환경](#2-2-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. inference 실행 가이드](#6-inference-실행-가이드)

## 1. 프로젝트 개요

* **Kvasir-SEG** 데이터셋에서 우수한 성능을 거둔 모델 중 하나로 **EffiSegNet** 이 있다. ([PapersWithCode 참고](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg))
  * [Kvasir-SEG 데이터셋](https://datasets.simula.no/kvasir-seg/) (상업적 사용을 위해서는 별도 허가 필요)
  * [EffiSegNet Official Code (PyTorch)](https://github.com/ivezakis/effisegnet)

| 모델            | mean Dice       | mIoU                |
|---------------|-----------------|---------------------|
| EffiSegNet-B5 | 0.9488 (rank 2) | **0.9065 (rank 1)** |
| EffiSegNet-B4 | 0.9483 (rank 3) | 0.9056 (rank 2)     |

* 본 프로젝트는 **EffiSegNet 으로 Kvasir-SEG 데이터셋을 학습** 시켰을 때,
  * **'오답'이 발생하는 부분** 을 찾고,
  * 그 오답의 원인과 해결 방법을 탐색 및 적용하여 성능을 향상시키는 것을 목표로 한다.

### 1-1. 성능 향상 방법

### 1-2. 성능 향상 결과

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Computer Vision
  * LLM (optional)
* 사용 기술

| 기술 분야           | 사용 기술      | 설명                         |
|-----------------|------------|----------------------------|
| Computer Vision | EffiSegNet | Kvasir-SEG 데이터셋에서의 병변 등 탐지 |

### 2-1. 관련 논문

### 2-2. 사용한 Python 라이브러리 및 시스템 환경

## 3. 프로젝트 일정

## 4. 프로젝트 상세 설명

## 5. 프로젝트 진행 중 이슈 및 해결 방법

## 6. inference 실행 가이드
 