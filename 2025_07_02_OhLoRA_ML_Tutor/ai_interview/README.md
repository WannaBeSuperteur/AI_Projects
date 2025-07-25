
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