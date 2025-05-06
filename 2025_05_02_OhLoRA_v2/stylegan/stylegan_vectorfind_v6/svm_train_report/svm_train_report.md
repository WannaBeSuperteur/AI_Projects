## 2025.05.06 18:27

* total synthesized samples = **60,000**
* use each property score top **300 (= 0.50%)** & bottom **300 (= 0.50%)** for SVM training

```
training SVM for eyes ...
accuracy          : 0.8083
recall    (large) : 0.8167
precision (large) : 0.8033
F1 score  (large) : 0.8099
recall    (small) : 0.8000
precision (small) : 0.8136
F1 score  (small) : 0.8067

training SVM for mouth ...
accuracy          : 0.8083
recall    (large) : 0.8333
precision (large) : 0.7937
F1 score  (large) : 0.8130
recall    (small) : 0.7833
precision (small) : 0.8246
F1 score  (small) : 0.8034

training SVM for pose ...
accuracy          : 0.9667
recall    (large) : 0.9833
precision (large) : 0.9516
F1 score  (large) : 0.9672
recall    (small) : 0.9500
precision (small) : 0.9828
F1 score  (small) : 0.9661
```

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