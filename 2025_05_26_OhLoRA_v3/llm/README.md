## 코드 실행 방법

모든 코드는 **먼저 아래 다운로드 경로 안내] (TBU) 및 해당 각 HuggingFace 링크에 있는 Model Card 에 나타난 저장 경로 (Save Path) 정보를 참고하여 모델 다운로드 후,** ```2025_05_26_OhLoRA_v3``` (프로젝트 메인 디렉토리) 에서 실행

### 1. LLM Fine-Tuning

지정된 경로에 해당 LLM 이 이미 존재하는 경우, Fine-Tuning 대신 **inference test 실행됨**

| LLM                                  | Fine-Tuning 코드 실행 방법                                                          |
|--------------------------------------|-------------------------------------------------------------------------------|
| 답변 메시지 ```output_message```          | ```llm/run_fine_tuning.py -llm_names kanana -output_cols output_message```    |
| 최근 대화 내용 요약 ```summary```            | ```llm/run_fine_tuning.py -llm_names polyglot -output_cols summary```         |
| 메모리 (사용자에 대해 기억 필요한 내용) ```memory``` | ```llm/run_fine_tuning.py -llm_names kanana -output_cols memory```            |
| 표정 및 고개 돌림 제어 ```eyes_mouth_pose```  | ```llm/run_fine_tuning.py -llm_names polyglot -output_cols eyes_mouth_pose``` |

### 2. Memory Mechanism (RAG-like concept)

* **Memory Mechanism (S-BERT)** 모델 실행 (해당 모델 없을 시, Training 먼저 실행)
  * ```python llm/run_memory_mechanism.py```

### 3. Ethics Mechanism

* **Ethics Mechanism (S-BERT)** 모델 실행 (해당 모델 없을 시, Training 먼저 실행)
  * ```python llm/run_ethics_mechanism.py```
