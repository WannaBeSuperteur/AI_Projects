
## 실험 결과

* 결론 요약

| 항목                | ```output_answered```<br>(사용자 답변 성공 여부 평가) | ```next_question```<br>(다음 질문 선택)       | LLM<br>(다음 질문 생성) |
|-------------------|--------------------------------------------|-----------------------------------------|-------------------|
| 사용 모델             | **S-BERT**<br>(roberta-base, 40 epochs)    | **S-BERT**<br>(roberta-base, 40 epochs) | Causal LLM        |
| 평가                | 실제 제품 적용이 가능할 정도로 **정확도가 충분히 높음**          | **정확도 매우 우수**                           |                   |
| 최종 정확도 (Accuracy) |                                            |                                         |                   |

* S-BERT 모델 테스트 결과
  * ```Accuracy```
    * S-BERT 모델이 **각 후보 답변과의 유사도 비교** 를 통해 결정한 **코사인 유사도가 가장 높은 답변** 이, 실제 ground truth 인 비율 
  * ```Mean_Diff```
    * 각 질문에 대해,
      * (코사인 유사도가 가장 높은 후보 답변) 과 (ground truth) 가 **일치하는** 경우
        * **(+1.0)** * {1위 후보 (= ground truth) 에 대한 코사인 유사도와, 2위 후보에 대한 코사인 유사도의 차이}
      * (코사인 유사도가 가장 높은 후보 답변) 과 (ground truth) 가 **서로 다른** 경우
        * **(-1.0)** * {실제 ground truth 답변에 대한 코사인 유사도와, 1위 후보 (= 오답) 에 대한 코사인 유사도의 차이}
  * **roberta-base, 40 epochs** 기준
  * 데이터셋에서 **정답** 매칭, **오답** 매칭 각각 ground-truth cos-sim 을 **+1.0, 0.0** 으로 지정

| 항목              | ```output_answered```<br>(사용자 답변 성공 여부 평가) | ```next_question```<br>(다음 질문 선택) |
|-----------------|--------------------------------------------|-----------------------------------|
| ```Accuracy```  | 95.0 % (76 / 80)                           | 100.0 % (80 / 80)                 |
| ```Mean_Diff``` | 0.9056                                     | 0.9897                            |

* LLM 테스트 결과
  * TBU 

### S-BERT 모델 테스트 결과

* 결론 요약
  * 학습 epoch 횟수가 클수록 대체로 성능이 향상됨

* 상세 분석
  * 1차 데이터셋 → 2차 데이터셋
    * **무엇인가 (예: 손실 함수) 의 정의를 묻는 질문** 에 대한 학습 데이터셋 증량
  * 2차 데이터셋 → 최종 데이터셋
    * ```output_answered``` S-BERT 모델 성능 향상을 위해, **손실 함수 관련 경험** 질문에 대한 사용자 입력 (사용자 답변) 부분에 **기본 경험 & 상세 경험** 을 구분

| 항목                                       | ```output_answered```<br>(사용자 답변 성공 여부 평가) | ```next_question```<br>(다음 질문 선택)    |
|------------------------------------------|--------------------------------------------|--------------------------------------|
| ```Accuracy```<br>(**1차, 2차, 최종** 데이터셋)  | ![image](../../images/250702_18.PNG)       | ![image](../../images/250702_17.PNG) |
| ```Mean_Diff```<br>(**1차, 2차, 최종** 데이터셋) | ![image](../../images/250702_20.PNG)       | ![image](../../images/250702_19.PNG) |
| MSE, MAE, Corr-coef (상관계수)               | ![image](../../images/250702_22.PNG)       | ![image](../../images/250702_21.PNG) |

### LLM 테스트 결과

## 코드 실행 방법

모든 코드는 **먼저 LLM 모델 정보 및 다운로드 경로 안내 (TBU) 및 해당 각 HuggingFace 링크에 있는 Model Card 에 나타난 저장 경로 (Save Path) 정보를 참고하여 모델 다운로드 후,** ```2025_07_02_OhLoRA_ML_Tutor``` (프로젝트 메인 디렉토리) 에서 실행

### S-BERT Training (사용자가 성공적으로 한 답변, 다음 질문 예측 각각)

* 지정된 경로에 해당 S-BERT 모델이 이미 존재하는 경우, Fine-Tuning 대신 **inference test 실행됨**

```python ai_interview/run_sbert.py```

### S-BERT 모델의 정확도 측정

* 입력
  * ```next_question``` S-BERT 모델 상세 테스트 결과 (```next_question_sbert/result/test_result_..._epoch_{epochs}.csv```)
  * ```output_answer``` S-BERT 모델 상세 테스트 결과 (```output_answer/result/test_result_..._epoch_{epochs}.csv```)
* 출력
  * ```next_question``` S-BERT 모델 정확도 측정 결과 (```next_question_sbert/result/test_accuracy.csv```)
  * ```output_answer``` S-BERT 모델 정확도 측정 결과 (```output_answer_sbert/result/test_accuracy.csv```)

```python ai_interview/run_compute_sbert_accuracy.py```