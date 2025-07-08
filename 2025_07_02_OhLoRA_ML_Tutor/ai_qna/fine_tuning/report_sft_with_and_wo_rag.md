
## 목차

* [1. 개요](#1-개요)
* [2. Fine-Tuning 설정](#2-fine-tuning-설정)
* [3. Fine-Tuning 결과](#3-fine-tuning-결과)

## 1. 개요

* 본 프로젝트에서 **ML 관련 질의응답 LLM** 을 Fine-Tuning 학습시킬 때, **RAG (Retrieval Augmented Generation)** 의 성능 향상 효과를 검증한다.
* [RAG (Retrieval Augmented Generation) 설명](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_RAG.md)

## 2. Fine-Tuning 설정

| 구분                    | 설정값<br>(with RAG concept)                                                                           | 설정값<br>(without RAG concept)                                                                    |
|-----------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| 사용 LLM (baseline)     | [Kanana-1.5-2.1B-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)     | [Kanana-1.5-2.1B-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505) |
| 초기 learning rate      | 0.0003 (= 3e-4)                                                                                     | 0.0003 (= 3e-4)                                                                                 |
| training batch size   | 2                                                                                                   | 2                                                                                               |
| LoRA Rank             | 64                                                                                                  | 64                                                                                              |
| LoRA alpha            | 16                                                                                                  | 16                                                                                              |
| LoRA dropout          | 0.05                                                                                                | 0.05                                                                                            |
| total training epochs | 20                                                                                                  | 20                                                                                              |
| 학습 데이터셋               | [SFT_**with**_RAG_concept.csv](../fine_tuning_dataset/SFT_with_RAG_concept.csv)<br>(train rows: 30) | [SFT_**wo**_RAG_concept.csv](../fine_tuning_dataset/SFT_wo_RAG_concept.csv)<br>(train rows: 30) |
| 답변 생성 temperature     | 0.6                                                                                                 | 0.6                                                                                             |

## 3. Fine-Tuning 결과

* Fine-Tuning Inference 테스트 결과 요약
  * **RAG concept 을 적용한 경우 정확도가 거의 100% 수준** 으로, 미 적용 시 (약 80% 추정) 보다 정확도가 훨씬 높음
  * [**(with RAG concept)** Fine-Tuning Inference test log](logs/kananai_sft_with_rag_inference_log_0.6.txt)
  * [**(without RAG concept)** Fine-Tuning Inference test log](logs/kananai_sft_wo_rag_inference_log_0.6.txt)

* Fine-Tuning Inference 테스트 결과 상세 **(정확도 부분)**
  * 각 질문 별 ```정답 생성 개수 / 전체 답변 생성 개수```
  * 종합 평가는 모든 질문 (총 30개) 에 대한 위 계산 결과값의 평균

| 사용자 질문                                                              | with RAG concept        | without RAG concept            |
|---------------------------------------------------------------------|-------------------------|--------------------------------|
| 종합 평가                                                               | **거의 완벽 (정확도 100.0 %)** | 일부 부정확한 답변 있음 **(정확도 80.0 %)** |
| ```코사인 유사도가 뭔지 궁금해```                                               | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Cosine Similarity 에 대해 알려줘!```                                   | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Cosine Similarity 는 어떻게 계산하지?```                                 | ✅ 4 / 4                 | ❌ 1 / 4                        |
| ```코사인 유사도 특징이 뭐야?```                                               | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```머신러닝 모델 평가하는 방법 알려줘```                                           | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Accuracy 계산 어떻게 하지?```                                           | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```True Positive 가 뭐야?```                                           | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```True Negative 는 뭔지 궁금해 그럼```                                     | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```False Positive 알려줘```                                            | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```False Negative (FN)```                                           | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Recall 계산법 어떻게 하는 거지?```                                         | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Recall 이 뭐야?```                                                  | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Precision 어떻게 계산하는지 정말 궁금해```                                    | ✅ 4 / 4                 | ❌ 0 / 4                        |
| ```Recall Precision 이거 헷갈리는데 잘 외우는 법 없을까?```                        | ✅ 4 / 4                 | ❌ 0 / 4                        |
| ```F1 Score 가 뭐야?```                                                | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```F1 Score 계산식 알려줘```                                              | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```F1 Score 는 왜 쓰는 거지?```                                           | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```IoU 라는 게 있는데 뭔지 궁금해```                                           | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```특이도는 뭐야?```                                                      | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```이진 분류에서는 어떤 성능평가 지표가 쓰이지?```                                     | ✅ 4 / 4                 | ❌ 0 / 4                        |
| ```PR-AUC, ROC-AUC 가 뭐야?```                                         | ✅ 4 / 4                 | ⚠ 3 / 4                        |
| ```PR-AUC 에 대해 아주 자세히 알려줘```                                        | ✅ 4 / 4                 | ❌ 0 / 4                        |
| ```ROC-AUC 는 뭔지 정말 궁금해```                                           | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```True Positive Rate 구하는 방법을 알려줘```                                | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```False Positive Rate 가 뭔지 정말 궁금하다```                              | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Confusion Matrix (혼동 행렬) 이 뭐야?```                                | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Confusion Matrix 는 그럼 어떻게 만들어?```                                | ✅ 4 / 4                 | ❌ 0 / 4                        |
| ```불량품 최소화해야 하면 불량이 Positive 일 때 Recall 이랑 Precision 중에 뭐가 더 좋지?``` | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```Normalization 정규화 그게 뭐야?```                                      | ✅ 4 / 4                 | ✅ 4 / 4                        |
| ```정규화가 뭐지? 정말 궁금해!```                                              | ✅ 4 / 4                 | ✅ 4 / 4                        |

