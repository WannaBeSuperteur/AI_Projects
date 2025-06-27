
## 코드 실행 방법

모든 코드는 **먼저 아래 다운로드 경로 안내 (TBU) 및 해당 각 HuggingFace 링크에 있는 Model Card 에 나타난 저장 경로 (Save Path) 정보를 참고하여 모델 다운로드 후,** ```2025_06_24_OhLoRA_v4``` (프로젝트 메인 디렉토리) 에서 실행

### LLM Fine-Tuning

지정된 경로에 해당 LLM 이 이미 존재하는 경우, Fine-Tuning 대신 **inference test 실행됨**

| LLM                         | Fine-Tuning 코드 실행 방법                |
|-----------------------------|-------------------------------------|
| 답변 메시지 ```output_message``` | ```python llm/run_fine_tuning.py``` |
