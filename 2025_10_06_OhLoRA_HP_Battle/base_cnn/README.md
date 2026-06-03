## Baseline CNN Model for Oh-LoRA Hyper-Param Battle

* Oh-LoRA 👱‍♀️ 하이퍼파라미터 튜닝 대결을 위한 **Baseline CNN Model (= 하이퍼파라미터 튜닝 대상)**
* Task : **Image Classification (Single-Label, Multi-Class)**
* ```test_accuracy_{dataset_name}.txt``` 는 **학습+검증 데이터 합산 2,000장 만으로 학습** 한 모델에 대한 정확도임

## 상세 정보

* [코드 파일](baseline_cnn.py)
* 테스트 정확도

| 데이터셋           | CIFAR-10 | Fashion MNIST | MNIST  |
|----------------|----------|---------------|--------|
| Accuracy (정확도) | 0.5794   | 0.8399        | 0.9783 |
