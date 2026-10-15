## 목차

* [1. 전체 이슈 요약](#1-전체-이슈-요약)
* [2. 이슈 상세](#2-이슈-상세)
  * [2-1. 확률 예측 모델 학습 시 Input contains NaN. 오류 발생 (해결 완료)](#2-1-확률-예측-모델-학습-시-input-contains-nan-오류-발생-해결-완료) 
  * [2-2. 확률 예측 모델 inference 결과가 학습/테스트한 모델과 다르게 나옴 (해결 완료)](#2-2-확률-예측-모델-inference-결과가-학습테스트한-모델과-다르게-나옴-해결-완료)

## 1. 전체 이슈 요약

| 이슈 분류 | 이슈                                                    | 날짜         | 심각성    | 상태    | 원인 (및 해결 방법)                                                                              | 교훈                                                         |
|-------|-------------------------------------------------------|------------|--------|-------|-------------------------------------------------------------------------------------------|------------------------------------------------------------|
| 확률 모델 | 확률 예측 모델 학습 시 `ValueError: Input contains NaN.` 오류 발생 | 2026.09.13 | **심각** | 해결 완료 | Optimizer 의 `eps=1e-8` 등이 `torch.float16` 이 0으로 처리됨 (추정) → `torch_dtype=torch.float32` 적용 | -                                                          |
| 확률 모델 | 확률 예측 모델 inference 결과가 학습/테스트한 모델과 다르게 나옴             | 2026.09.25 | **심각** | 해결 완료 | 모델 checkpoint 누락 해결                                                                       | 딥러닝 모델 재현성 (random seed 고정) 은 **재현이 버그 해결에 중요하므로 심각한 문제임** |

## 2. 이슈 상세

### 2-1. 확률 예측 모델 학습 시 Input contains NaN. 오류 발생 (해결 완료)

**1. 문제 상황**

* 임베딩 기반 확률 모델 학습 중, 다음과 같이 `ValueError: Input contains NaN.` 오류 발생

```
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 4288.95it/s]
Traceback (most recent call last):
  File "C:\Users\20151\Documents\AI_Projects\2026_08_28_OhLoRA_Code_Assistant\embedding_model\train_model.py", line 531, in <module>
    train_probability_predictor(model_path, dataset_path, task_name)
  File "C:\Users\20151\Documents\AI_Projects\2026_08_28_OhLoRA_Code_Assistant\embedding_model\train_model.py", line 328, in train_probability_predictor
    trainer.run()
  File "C:\Users\20151\Documents\AI_Projects\2026_08_28_OhLoRA_Code_Assistant\embedding_model\train_model.py", line 300, in run
    self._run_all_process()
  File "C:\Users\20151\Documents\AI_Projects\2026_08_28_OhLoRA_Code_Assistant\embedding_model\train_model.py", line 232, in _run_all_process
    valid_mse, valid_mae, valid_loss = self._run_validation_or_test(model=self.predictor,
  File "C:\Users\20151\Documents\AI_Projects\2026_08_28_OhLoRA_Code_Assistant\embedding_model\train_model.py", line 187, in _run_validation_or_test
    val_mse_batch = sklearn.metrics.mean_squared_error(preds, prob_labels)
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\sklearn\utils\_param_validation.py", line 214, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\sklearn\metrics\_regression.py", line 474, in mean_squared_error
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\sklearn\metrics\_regression.py", line 100, in _check_reg_targets
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\sklearn\utils\validation.py", line 957, in check_array
    _assert_all_finite(
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\sklearn\utils\validation.py", line 122, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\sklearn\utils\validation.py", line 171, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input contains NaN.
```

**2. 문제 원인 및 해결 방법**

* 문제 원인 (추정)
  * 다음 부분에서, `torch16` 설정으로 인해 Optimizer 의 `eps=1e-8` 값이 0으로 처리됨

```python
def train_probability_predictor(model_path: str, dataset_path: str, task_name: str):
    """train text embedding probability predictor."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
```

* 해결 방법
  * `torch.float16` 을 `torch.float32` 로 다음과 같이 수정

```python
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
```

* 시도했으나 실패한 해결 방법
  * 다음과 같이 특정 모델에 대해, 첫번째 임베딩 `outputs.last_hidden_state[:, 0]` 만을 사용하도록 하는 방법

```python
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.base_model_path == GTE_MODERNBERT_BASE:
            emb = outputs.last_hidden_state[:, 0].float()  # 첫번째 임베딩만 사용
        else:
            emb = mean_pooling(outputs, attention_mask)

        prob = self.final_linear(emb)
        return prob
```

### 2-2. 확률 예측 모델 inference 결과가 학습/테스트한 모델과 다르게 나옴 (해결 완료)

**1. 문제 상황**

* 확률 예측 모델을 실제 코드를 대상으로 inference 시, 해당 모델 학습 시의 **학습 완료 후 테스트 결과와 다른 값** 이 출력됨

```
==== RULE : 01_unnecessary_prints ====
파일: test_cases\tc01_01_basics.py
 - 함수: (최상위 레벨)
   - line 18 에 있는 logger.info("3335555"), [AI] prob=0.44991493225097656

파일: test_cases\tc01_01_basics_2.py
 - 함수: (최상위 레벨)
   - line 20 에 있는 print(IMPORT_NUM), [AI] prob=0.6139438152313232
   - line 20 에 있는 print(IMPORT_NUM)
   - line 22 에 있는 print(imported_value), [AI] prob=0.6374119520187378
   - line 22 에 있는 print(imported_value)

파일: test_cases\tc01_02.py
 - 함수: (최상위 레벨)
   - line 5 에 있는 logger.warning("server error detected"), [AI] prob=0.6679722666740417
   - line 5 에 있는 logger.warning("server error detected")
   - line 6 에 있는 logger.error("server fault"), [AI] prob=0.5971484780311584
```

**2. 문제 원인 및 해결 방법**

* 문제 원인
  * inference 에서 **모델 checkpoint loading 하는 부분이 누락** 되어 있었음

* 해결 방법
  * 모델 checkpoint loading 하는 부분 적용 (누락 해결)
  * 해결을 위한 사전 작업으로, **재현성 보장한 후 모든 모델 재 학습 (재현성 체크 완료)**

**3. 교훈**

* 딥러닝 모델 재현성 (random seed 고정) 은 **재현이 버그 해결에 중요하므로 심각한 문제임**
* 재현성 설정 누락 등으로 재현이 안 될 시, **모델을 처음부터 재 학습 (많은 GPU 자원, 시간 필요) 해야 함!!**


