
## 목차

* [1. LLM 후보](#1-llm-후보)
  * [1-1. LLM 후보 선택 이유](#1-1-llm-후보-선택-이유)
* [2. LLM-as-a judge 평가 결과](#2-llm-as-a-judge-평가-결과)

## 1. LLM 후보

* 고려 사항
  * 현재 사용 가능한 GPU: **2 x Quadro M6000 (12GB)** 

* 선정 기준
  * 다음 2가지를 모두 만족시키는 모델 선정
    * 카카오 `kanana`, LG `EXAONE`, 네이버 `HyperCLOVA-X`, KT `Mi:dm` 에 속한 모델
    * **1.5B - 3B** 파라미터 모델
  * LLM-as-a-judge 방법으로 최종 평가하여, 최종 1개 선정 (ChatGPT 이용) 

* 선정 결과

| 후보 모델                                                   | HuggingFace Link                                                                                 | 개발사 | 파라미터 개수 | 최종 평가 결과                                                                                  | 최종 선정 여부 |
|---------------------------------------------------------|--------------------------------------------------------------------------------------------------|-----|---------|-------------------------------------------------------------------------------------------|----------|
| `kakaocorp/kanana-2-3b-instruct`                        | [HuggingFace Link](https://huggingface.co/kakaocorp/kanana-2-3b-instruct)                        | 카카오 | 3.0 B   | 1000 / 1000                                                                               | ✅        |
| `kakaocorp/kanana-1.5-2.1b-instruct-2505`               | [HuggingFace Link](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)               | 카카오 | 2.1 B   | ❌ **`transformers==5.17.0` 실행 불가**<br>(hidden size = 1792 로 attention head 개수 24의 배수가 아님) |          |
| `K-intelligence/Midm-2.0-Mini-Instruct`                 | [HuggingFace Link](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct)                 | KT  | 2.0 B   | 990 / 1000                                                                                |          |
| `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | [HuggingFace Link](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B) | 네이버 | 1.5 B   | 1000 / 1000                                                                               |          |

### 1-1. LLM 후보 선택 이유

* 선택한 후보
  * `kakaocorp/kanana-2-3b-instruct`

* 선택 이유
  * 최종 inference 결과 100개에 대한 평가 (LLM-as-a-judge 방식) 기준, 다음 2개의 모델이 만점
    * `kakaocorp/kanana-2-3b-instruct`
    * `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`
  * 이 중 `HyperCLOVAX-SEED-Text-Instruct-1.5B` 는 라이선스 조항에 따른 제한이 비교적 크기 때문에, **라이선스 제한이 적은 `kakaocorp/kanana-2-3b-instruct` 를 최종 선정**
    * 라이선스는 **모델 사용 조건** 이므로, 추론 시간, 파라미터 개수 (경량성) 보다 우선한다고 판단
    * [`HyperCLOVAX-SEED-Text-Instruct-1.5B` 의 라이선스 조항](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B/blob/main/LICENSE)

## 2. LLM-as-a judge 평가 결과

* 평가 방법
  * ChatGPT Work 의 GPT-5.6 Terra 최대

* 평가 프롬프트

```
위 3개 파일에서, epoch가 "final_inference"인 마지막 100개의 행을 보고, 'prompt' 컬럼 값의 5개 항목 중 마지막 항목과 'llm_answer' 컬럼 값의 관련성을 각각 평가해 줘. (관련성 있음+해결 방법 적절히 제시 = 10점, 관련성 있음+해결 방법 미 제시 = 9점, 관련성 없음 = 0점) 그리고 평가 결과를 csv 파일로 저장해 줘. (개별 llm_answer 에 대한 평가 결과 포함) 단, llm_answer의 마지막에 있는 '답변 종료'는 무시한다.
```

* 평가 결과
  * [상세 평가 결과](final_inference_llm_answer_evaluation.csv)

| 모델                                                      | 평가 점수       | 0점 | 9점 | 10점 |
|---------------------------------------------------------|-------------|----|----|-----|
| `kakaocorp/kanana-2-3b-instruct`                        | 1000 / 1000 | 0  | 0  | 100 |
| `K-intelligence/Midm-2.0-Mini-Instruct`                 | 990 / 1000  | 1  | 0  | 99  |
| `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | 1000 / 1000 | 0  | 0  | 100 |
