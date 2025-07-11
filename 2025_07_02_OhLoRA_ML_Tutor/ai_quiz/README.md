## 코드 실행 방법

모든 코드는 **먼저 LLM 모델 정보 및 다운로드 경로 안내 (TBU) 및 해당 각 HuggingFace 링크에 있는 Model Card 에 나타난 저장 경로 (Save Path) 정보를 참고하여 모델 다운로드 후,** ```2025_06_24_OhLoRA_v4``` (프로젝트 메인 디렉토리) 에서 실행

### LLM Fine-Tuning (해설용)

지정된 경로에 해당 LLM 이 이미 존재하는 경우, Fine-Tuning 대신 **inference test 실행됨**

```python ai_quiz/run_llm_fine_tuning.py```

### S-BERT Training (사용자 답변 채점용)

지정된 경로에 해당 S-BERT 모델이 이미 존재하는 경우, Fine-Tuning 대신 **inference test 실행됨**

```python ai_quiz/run_sbert.py```