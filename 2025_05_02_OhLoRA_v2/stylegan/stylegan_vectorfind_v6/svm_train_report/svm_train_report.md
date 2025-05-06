## 2025.05.06 17:45

* total synthesized samples = **12,000**
* use each property score top **100 (= 0.83%)** & bottom **100 (= 0.83%)** for SVM training

```
training SVM for eyes ...
accuracy          : 0.8000
recall    (large) : 0.7500
precision (large) : 0.8333
F1 score  (large) : 0.7895
recall    (small) : 0.8500
precision (small) : 0.7727
F1 score  (small) : 0.8095

training SVM for mouth ...
accuracy          : 0.6750
recall    (large) : 0.7500
precision (large) : 0.6522
F1 score  (large) : 0.6977
recall    (small) : 0.6000
precision (small) : 0.7059
F1 score  (small) : 0.6486

training SVM for pose ...
accuracy          : 0.8750
recall    (large) : 0.9000
precision (large) : 0.8571
F1 score  (large) : 0.8780
recall    (small) : 0.8500
precision (small) : 0.8947
F1 score  (small) : 0.8718
```