
## LLM 후보

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
| `kakaocorp/kanana-2-3b-instruct`                        | [HuggingFace Link](https://huggingface.co/kakaocorp/kanana-2-3b-instruct)                        | 카카오 | 3.0 B   |                                                                                           |          |
| `kakaocorp/kanana-1.5-2.1b-instruct-2505`               | [HuggingFace Link](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)               | 카카오 | 2.1 B   | ❌ **`transformers==5.17.0` 실행 불가**<br>(hidden size = 1792 로 attention head 개수 24의 배수가 아님) |          |
| `K-intelligence/Midm-2.0-Mini-Instruct`                 | [HuggingFace Link](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct)                 | KT  | 2.0 B   |                                                                                           |          |
| `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | [HuggingFace Link](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B) | 네이버 | 1.5 B   |                                                                                           |          |
