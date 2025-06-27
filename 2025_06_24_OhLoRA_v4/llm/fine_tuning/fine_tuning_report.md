## 1. LLM Fine-Tuning 학습 설정

* 사용 LLM
  * [Kanana-1.5-2.1B-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505) 를 baseline 모델로 채택 
* 학습 기본 설정
  * learning_rate : **0.0003**
  * training batch size : **2**
  * LoRA Rank : **16**
  * LoRA alpha : **16**
  * LoRA dropout : **0.05**
  * total training epochs : **5 epochs (✅ 최종 채택) / 10 epochs**
* 학습 데이터셋
  * [OhLoRA_fine_tuning_v4.csv](../fine_tuning_dataset/OhLoRA_fine_tuning_v4.csv)
  * train rows : **632**
  * validation rows : **120**
* 답변 생성 기본 설정
  * temperature : **0.6** 

## 2. LLM Fine-Tuning 결과 요약

| 평가 항목       | 5 epochs                                           | 10 epochs                           |
|-------------|----------------------------------------------------|-------------------------------------|
| 숫자로 시작하는 답변 | **거의 없음**                                          | 상대적으로 많음                            |
| 기본 답변 성능    | **대체로 좋음**                                         | 학습 데이터의 내용을 맥락과 무관하게 출력하는 경우가 종종 있음 |
| 윤리적 답변      | **비교적 우수**<br>(악성 사용자의 '유도 심문'에 대한 거부 or 회피 경향 높음) | 비교적 윤리에 어긋나는 답변이 많음                 |
| 최종 성능 비교    | **비교적 우수**                                         | 비교적 낮음                              |

* 10 epochs 에서, **학습 후반부로 갈수록 숫자로 시작하는 답변이 더 많이 생성** 되는 경향이 있음

## 3. 최종 출력 로그

* [5 epochs 최종 로그](logs/kananai_output_message_5epochs_inference_log_0.6.txt)
* [10 epochs 최종 로그](logs/kananai_output_message_10epochs_inference_log_0.6.txt)
