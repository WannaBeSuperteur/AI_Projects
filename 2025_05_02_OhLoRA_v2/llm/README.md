## 코드 실행 방법

모든 코드는 ```2025_05_02_OhLoRA_v2``` (프로젝트 메인 디렉토리) 에서 실행

* **Polyglot-Ko 1.3B** Fine-Tuned 모델 실행 (해당 모델 없을 시, Fine-Tuning 먼저 실행)
  * option 1 
    * ```python llm/run_fine_tuning.py -llm_name polyglot```
  * option 2
    * ```python llm/run_fine_tuning.py```

### 모델 다운로드 경로

| 모델 이름                  | 원본 모델                                                                                | Fine-Tuned LLM<br>(for OhLoRA-v2 👱‍♀️) |
|------------------------|--------------------------------------------------------------------------------------|-----------------------------------------|
| ```Polyglot-Ko 1.3B``` | [EleutherAI HuggingFace](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)         | TBU                                     |
| ```KoreanLM 1.5B```    | [Quantum AI HuggingFace](https://huggingface.co/quantumaikr/KoreanLM-1.5b/tree/main) | ❌ 학습 실패 **(참고: TBU)**                   |