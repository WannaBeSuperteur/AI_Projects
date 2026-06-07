# Oh-LoRA HP Battle QA List

## 목차

* [1. 기본 사항](#1-기본-사항)
* [2. 기본 기능](#2-기본-기능)
* [3. 기능적 메커니즘](#3-기능적-메커니즘)

## 1. 기본 사항

* 본 문서는 **Oh-LoRA 👱‍♀️ (오로라) HP Battle** 에 대한 체계적인 검증을 위한 QA List 이다.
* 1차 검사 통과 시, **1차 검사 결과** 를 최종 결과로 하고 **최종 검사 결과** 란은 비운다.
* 검사 결과는 이모지 (✅ or ❌) 로 나타낸다.

## 2. 기본 기능

| 번호 | 검사 항목               | 1차 검사 결과 | 버그 수정 | 최종 검사 결과 |
|----|---------------------|----------|-------|----------|
| 1  | 실행 시 정상 동작          | ✅        |       |          |
| 2  | 실행 중 memory leak 없음 | ✅        |       |          |

## 3. 기능적 메커니즘

| 번호 | 검사 항목                         | 1차 검사 결과 | 버그 수정 | 최종 검사 결과 |
|----|-------------------------------|----------|-------|----------|
| 3  | sub-dataset 수집                | ✅        |       |          |
| 4  | 각 데이터셋 별 Oh-LoRA 선공           | ✅        |       |          |
| 5  | 인간 예측 실시 (1차)                 | ✅        |       |          |
| 6  | 인간 예측 실시 (2차)                 | ✅        |       |          |
| 7  | 최종 결과 표시 (승리, 패배)             | ✅        |       |          |
| 8  | 최종 점수 표시 (인간 1차, 2차, Oh-LoRA) | ✅        |       |          |
| 9  | Oh-LoRA 가 도출한 하이퍼파라미터 표시      | ✅        |       |          |
| 10 | Oh-LoRA 의 LLM 출력 표시           | ✅        |       |          |

## 4. QA 로그

* PyCharm의 버그로 인해 일부 라인이 누락되어 있을 수 있습니다.

<details><summary>상세 QA 로그 [ 펼치기 / 접기 ]</summary>

```
새로운 크로스 플랫폼 PowerShell 사용 https://aka.ms/pscore6

(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects> cd 2025_10_06_OhLoRA_HP_Battle
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle> git status
On branch main
Your branch is up to date with 'origin/main'.

On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle>
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle>
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle>
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle>
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle>
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle>
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle> python run_battle_vs_human.py


[ 데이터셋 이름 : cifar_10 ]
rejected distribution: train=[3, 3, 2, 1], test=[]
rejected distribution: train=[393, 380, 349, 320, 319], test=[86, 76, 65, 64, 64]
rejected distribution: train=[2, 1, 1], test=[1]
rejected distribution: train=[41, 28, 24, 20, 20], test=[6, 6, 4, 4, 4]
rejected distribution: train=[19, 15, 4, 3, 2], test=[3, 2]
rejected distribution: train=[], test=[]
rejected distribution: train=[598, 449, 375, 369, 363], test=[123, 97, 77, 68, 66]
rejected distribution: train=[38, 36, 21, 20, 19], test=[10, 5, 4, 4, 3]
rejected distribution: train=[71, 67, 61, 56, 31], test=[14, 14, 14, 7, 4]
rejected distribution: train=[11, 10, 6, 2, 2], test=[4, 3]
rejected distribution: train=[], test=[]
rejected distribution: train=[1, 1, 1], test=[]
rejected distribution: train=[91, 83, 66, 56, 49], test=[17, 15, 13, 9, 4]
rejected distribution: train=[2, 1, 1], test=[]
rejected distribution: train=[31, 29, 18, 15, 15], test=[9, 7, 6, 3, 2]
rejected distribution: train=[98, 67, 36, 14, 8], test=[15, 11, 11, 3, 2]
rejected distribution: train=[18, 11, 8, 6, 5], test=[3, 2, 1, 1]
rejected distribution: train=[27, 21, 11, 8, 7], test=[3, 3, 2, 1]
rejected distribution: train=[187, 105, 104, 65, 19], test=[43, 24, 19, 16, 2]
rejected distribution: train=[25, 16, 12, 9, 8], test=[7, 4, 3, 2, 2]
rejected distribution: train=[334, 187, 156, 146, 82], test=[72, 39, 35, 28, 19]
rejected distribution: train=[113, 83, 49, 36, 10], test=[22, 20, 10, 4, 1]
rejected distribution: train=[364, 274, 55, 39, 38], test=[57, 53, 10, 9, 7]
rejected distribution: train=[1, 1], test=[]
rejected distribution: train=[57, 57, 48, 42, 39], test=[9, 8, 8, 6, 6]
rejected distribution: train=[23, 23, 5, 4, 2], test=[4, 3, 1, 1]
rejected distribution: train=[4, 2, 2, 1, 1], test=[]
rejected distribution: train=[93, 81, 56, 53, 50], test=[21, 15, 10, 9, 6]
rejected distribution: train=[42, 24, 11, 2, 2], test=[5, 2]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[7, 5, 1], test=[1, 1]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1058/1058 [00:02<00:00, 439.12it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 187/187 [00:01<00:00, 94.93it/s] 
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 283/283 [00:03<00:00, 91.77it/s] 


=== Oh-LoRA 👱‍♀️ 선공 예측 ===
epoch : 0, train_loss : 2.1726, val_loss : 2.0692
epoch : 1, train_loss : 2.0510, val_loss : 2.0264
epoch : 2, train_loss : 2.0254, val_loss : 2.0111
epoch : 3, train_loss : 2.0022, val_loss : 1.9587
epoch : 4, train_loss : 1.9714, val_loss : 1.9461
epoch : 5, train_loss : 1.9557, val_loss : 1.9611
epoch : 6, train_loss : 1.9497, val_loss : 1.9272
epoch : 7, train_loss : 1.9382, val_loss : 1.9292
epoch : 8, train_loss : 1.9284, val_loss : 1.9204
epoch : 9, train_loss : 1.9255, val_loss : 1.9252
epoch : 10, train_loss : 1.9182, val_loss : 1.9128
epoch : 11, train_loss : 1.9181, val_loss : 1.9170
epoch : 12, train_loss : 1.9132, val_loss : 1.9082
epoch : 13, train_loss : 1.9110, val_loss : 1.9106
epoch : 14, train_loss : 1.9053, val_loss : 1.9199
epoch : 15, train_loss : 1.8972, val_loss : 1.9166
accuracy : 0.5018, f1_score: (macro: 0.4878, micro: 0.5018)


=== 인간 1차 예측 ===

[ SYSTEM MESSAGE ]
데이터셋이 battle_dataset/ 경로에 저장되었습니다.
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}

(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)

[ 인간의 1차 하이퍼파라미터 : {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.0, 'lr': 0.000725, 'activation_func': 'leaky_relu', 'optimizer': 'adamw', 'scheduler': 'exp_95'} ]
epoch : 0, train_loss : 2.1547, val_loss : 2.0412
epoch : 1, train_loss : 2.0481, val_loss : 2.0386
epoch : 2, train_loss : 2.0496, val_loss : 2.0438
epoch : 3, train_loss : 2.0507, val_loss : 2.0387
epoch : 4, train_loss : 2.0629, val_loss : 2.0386
epoch : 5, train_loss : 2.0532, val_loss : 2.0382
accuracy : 0.417, f1_score: (macro: 0.2432, micro: 0.417)
[ 인간의 Macro F1 Score = 0.2432 ]


=== 인간 2차 예측 ===

[ SYSTEM MESSAGE ]
데이터셋이 battle_dataset/ 경로에 저장되었습니다.
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}

(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)

[ 인간의 2차 하이퍼파라미터 : {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.0, 'lr': 0.000612, 'activation_func': 'leaky_relu', 'optimizer': 'adamw', 'scheduler': 'exp_95'} ]
epoch : 0, train_loss : 2.1033, val_loss : 2.0354
epoch : 1, train_loss : 2.0537, val_loss : 2.1338
epoch : 2, train_loss : 2.0401, val_loss : 2.0254
epoch : 3, train_loss : 2.0356, val_loss : 2.0399
epoch : 4, train_loss : 2.0152, val_loss : 1.9861
epoch : 5, train_loss : 1.9975, val_loss : 1.9855
epoch : 6, train_loss : 2.0009, val_loss : 1.9960
epoch : 7, train_loss : 1.9992, val_loss : 1.9720
epoch : 8, train_loss : 1.9978, val_loss : 1.9887
epoch : 9, train_loss : 1.9849, val_loss : 1.9662
epoch : 10, train_loss : 1.9832, val_loss : 1.9940
epoch : 11, train_loss : 1.9813, val_loss : 1.9713
epoch : 12, train_loss : 1.9775, val_loss : 1.9612
epoch : 13, train_loss : 1.9689, val_loss : 1.9702
epoch : 14, train_loss : 1.9675, val_loss : 1.9685
epoch : 15, train_loss : 1.9627, val_loss : 1.9728
accuracy : 0.4806, f1_score: (macro: 0.4186, micro: 0.4806)
[ 인간의 Macro F1 Score = 0.4186 ]
[ Oh-LoRA 👱‍♀️ 의 Macro F1 Score = 0.4878 ] (예측: 0.49076077342033386)
[ Oh-LoRA 👱‍♀️ 의 하이퍼파라미터 = {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.6, 'lr': 0.00022258610759949617, 'activation_func': 'relu', 'optimizer': 'adam', 'scheduler': 'exp_90'} ]
[ 최종 결과 : Oh-LoRA 👱‍♀️ 의 승리 ]

 ==== 상세 점수 ====
Human 1st : 0.2432
Human 2nd : 0.4186
Oh-LoRA   : 0.4878
=================


Oh-LoRA 👱‍♀️ :  와 이건 좀 심각한데? 🤣 그래도 포기하지 말고 하이퍼파라미터를 다시 잘 설정해 보자! 파이팅!



[ 데이터셋 이름 : fashion_mnist ]
rejected distribution: train=[], test=[]
rejected distribution: train=[28, 5, 3], test=[4, 1]
rejected distribution: train=[130, 73, 68, 12, 5], test=[26, 13, 6, 2, 2]
rejected distribution: train=[], test=[]
rejected distribution: train=[823, 519, 506, 322, 177], test=[134, 92, 89, 51, 36]
rejected distribution: train=[4, 3, 2, 2, 2], test=[2, 2, 1]
rejected distribution: train=[657, 642, 317, 237, 115], test=[137, 110, 48, 44, 18]
rejected distribution: train=[1557, 961, 745, 706, 657], test=[238, 151, 125, 109, 97]
rejected distribution: train=[], test=[]
rejected distribution: train=[429, 278, 134, 116, 32], test=[65, 47, 23, 22, 4]
rejected distribution: train=[], test=[]
rejected distribution: train=[276, 210, 162, 79, 40], test=[37, 36, 26, 14, 10]
rejected distribution: train=[], test=[]
rejected distribution: train=[4535, 4269, 4243, 3368, 3290], test=[766, 720, 707, 553, 547]
rejected distribution: train=[2571, 876, 240, 120, 68], test=[415, 142, 38, 11, 8]
rejected distribution: train=[1340, 468, 354, 185, 150], test=[222, 76, 71, 42, 31]
rejected distribution: train=[4136, 3181, 931, 900, 533], test=[710, 502, 148, 127, 69]
rejected distribution: train=[101, 94, 88, 83, 64], test=[19, 12, 12, 11, 9]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[1501, 1436, 1278, 1222, 885], test=[258, 222, 213, 211, 154]
rejected distribution: train=[], test=[]
rejected distribution: train=[326, 215, 185, 156, 138], test=[58, 45, 23, 22, 21]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[521, 422, 216, 130, 72], test=[86, 77, 27, 19, 19]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[155, 92, 76, 16, 8], test=[28, 11, 7, 2, 1]
rejected distribution: train=[51, 26, 11, 1, 1], test=[6, 3, 3]
rejected distribution: train=[11, 7, 6, 3, 1], test=[2, 1]
rejected distribution: train=[1139, 725, 577, 262, 250], test=[216, 124, 109, 45, 44]
rejected distribution: train=[7, 5, 5, 2, 1], test=[1]
rejected distribution: train=[19, 10, 6, 1, 1], test=[5]
rejected distribution: train=[723, 642, 516, 344, 271], test=[147, 84, 80, 50, 41]
rejected distribution: train=[248, 142, 126, 69, 22], test=[44, 28, 22, 14, 3]
rejected distribution: train=[3733, 2143, 1913, 1832, 930], test=[648, 377, 338, 309, 151]
rejected distribution: train=[1339, 668, 259, 146, 49], test=[204, 125, 35, 24, 13]
rejected distribution: train=[1099, 659, 648, 517, 261], test=[185, 119, 102, 95, 50]
rejected distribution: train=[3011, 2923, 2675, 2087, 1581], test=[513, 492, 465, 333, 250]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[330, 183, 123, 60, 44], test=[58, 29, 21, 8, 5]
C:\Users\20151\Documents\AI_Projects\venv\lib\site-packages\torch\utils\data\dataset.py:473: UserWarning: Length of split at index 0 is 0. This might result in an empty dataset.
  warnings.warn(
rejected distribution: train=[1], test=[]
rejected distribution: train=[5, 3], test=[]
rejected distribution: train=[140, 9, 3, 2, 1], test=[27, 1]
rejected distribution: train=[3416, 3113, 2755, 2582, 2274], test=[551, 519, 473, 418, 353]
rejected distribution: train=[62, 32, 19, 7, 4], test=[6, 4, 2, 2, 1]
rejected distribution: train=[], test=[]
rejected distribution: train=[3387, 2388, 1439, 1298, 1072], test=[595, 415, 252, 241, 164]
rejected distribution: train=[], test=[]
rejected distribution: train=[355, 351, 350, 251, 221], test=[56, 51, 44, 43, 37]
rejected distribution: train=[401, 290, 259, 208, 127], test=[76, 43, 40, 36, 16]
rejected distribution: train=[2924, 2219, 2077, 2038, 1935], test=[463, 367, 356, 354, 341]
rejected distribution: train=[672, 610, 394, 186, 146], test=[109, 97, 63, 30, 27]
rejected distribution: train=[12, 12, 4, 4, 3], test=[3, 3, 2, 1]
rejected distribution: train=[2496, 1852, 1403, 1373, 1246], test=[401, 347, 222, 216, 209]
rejected distribution: train=[474, 361, 265, 104, 42], test=[72, 59, 48, 24, 8]
rejected distribution: train=[4914, 4174, 2167, 2136, 2102], test=[819, 683, 401, 344, 333]
rejected distribution: train=[944, 600, 553, 465, 204], test=[146, 107, 98, 70, 38]
rejected distribution: train=[2888, 2736, 2484, 2079, 1972], test=[490, 465, 419, 312, 310]
rejected distribution: train=[2, 1, 1], test=[1]
rejected distribution: train=[200, 156, 96, 51, 18], test=[29, 22, 14, 6, 1]
rejected distribution: train=[], test=[]
rejected distribution: train=[14, 6, 3, 3, 2], test=[2, 1, 1]
rejected distribution: train=[73, 55, 47, 31, 28], test=[11, 8, 8, 3, 3]
rejected distribution: train=[1048, 739, 629, 419, 266], test=[170, 135, 117, 71, 54]
rejected distribution: train=[284, 225, 158, 56, 33], test=[51, 28, 23, 14, 5]
rejected distribution: train=[128, 118, 52, 29, 23], test=[21, 19, 15, 4, 3]
rejected distribution: train=[7, 6, 5, 5, 5], test=[3, 1, 1, 1]
rejected distribution: train=[], test=[]
rejected distribution: train=[82, 78, 31, 11, 10], test=[10, 6, 3, 2, 1]
rejected distribution: train=[302, 223, 101, 81, 25], test=[52, 36, 14, 12, 6]
rejected distribution: train=[], test=[]
rejected distribution: train=[4487, 3668, 3295, 2930, 1683], test=[766, 636, 554, 510, 247]
rejected distribution: train=[], test=[]
rejected distribution: train=[652, 634, 392, 373, 329], test=[127, 80, 64, 58, 57]
rejected distribution: train=[32, 31, 8, 5, 4], test=[6, 3]
rejected distribution: train=[], test=[]
rejected distribution: train=[49, 40, 27, 26, 8], test=[7, 4, 2, 1, 1]
rejected distribution: train=[912, 689, 553, 458, 368], test=[169, 108, 86, 73, 61]
rejected distribution: train=[327, 322, 321, 301, 242], test=[53, 51, 49, 46, 44]
rejected distribution: train=[1266, 955, 534, 297, 159], test=[210, 158, 76, 57, 19]
rejected distribution: train=[4142, 2910, 2012, 1798, 1739], test=[720, 470, 356, 271, 269]
rejected distribution: train=[], test=[]
rejected distribution: train=[4, 3, 3, 2, 1], test=[1, 1]
rejected distribution: train=[245, 171, 168, 68, 33], test=[45, 30, 22, 9, 1]
rejected distribution: train=[1255, 1063, 1043, 782, 779], test=[230, 178, 158, 131, 121]
rejected distribution: train=[33, 21, 14, 13, 5], test=[7, 5, 5, 3, 2]
rejected distribution: train=[764, 530, 519, 479, 476], test=[118, 95, 88, 87, 85]
rejected distribution: train=[47, 43, 22, 17, 14], test=[6, 4, 3, 3, 3]
rejected distribution: train=[206, 193, 177, 177, 114], test=[42, 38, 37, 35, 18]
rejected distribution: train=[3, 1], test=[]
rejected distribution: train=[2451, 2215, 2149, 1870, 1648], test=[405, 392, 368, 293, 290]
rejected distribution: train=[5, 3, 1, 1, 1], test=[3, 1]
rejected distribution: train=[4307, 3823, 3512, 3161, 2952], test=[683, 643, 582, 540, 485]
rejected distribution: train=[5186, 4805, 4700, 3972, 3634], test=[867, 805, 787, 666, 608]
rejected distribution: train=[], test=[]
rejected distribution: train=[18, 17, 9, 4, 2], test=[3, 2, 2]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[158, 83, 52, 49, 27], test=[24, 17, 10, 10, 8]
rejected distribution: train=[709, 427, 421, 366, 127], test=[112, 77, 70, 69, 29]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[405, 359, 174, 167, 151], test=[84, 44, 36, 23, 23]
rejected distribution: train=[], test=[]
rejected distribution: train=[1953, 1810, 1274, 1095, 823], test=[332, 321, 205, 182, 153]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[198, 176, 106, 56, 31], test=[27, 24, 9, 7, 2]
rejected distribution: train=[285, 170, 167, 140, 126], test=[52, 31, 29, 21, 20]
rejected distribution: train=[4802, 4132, 3845, 2835, 2268], test=[793, 708, 658, 463, 397]
rejected distribution: train=[1725, 341, 121, 108, 103], test=[274, 60, 22, 19, 18]
rejected distribution: train=[95, 79, 72, 45, 43], test=[8, 7, 7, 4, 2]
rejected distribution: train=[1557, 1411, 1135, 620, 613], test=[257, 210, 181, 108, 90]
rejected distribution: train=[880, 793, 693, 608, 304], test=[153, 123, 96, 90, 47]
rejected distribution: train=[], test=[]
rejected distribution: train=[594, 492, 333, 269, 216], test=[119, 61, 54, 53, 33]
rejected distribution: train=[6], test=[2]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1127/1127 [00:02<00:00, 466.36it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:02<00:00, 93.46it/s] 
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 210/210 [00:02<00:00, 101.89it/s] 


=== Oh-LoRA 👱‍♀️ 선공 예측 ===
epoch : 0, train_loss : 2.2093, val_loss : 2.1685
epoch : 1, train_loss : 2.0981, val_loss : 1.8944
epoch : 2, train_loss : 1.9056, val_loss : 1.8563
epoch : 3, train_loss : 1.8748, val_loss : 1.8396
epoch : 4, train_loss : 1.8183, val_loss : 1.8325
epoch : 5, train_loss : 1.8051, val_loss : 1.7955
epoch : 6, train_loss : 1.7888, val_loss : 1.7881
epoch : 7, train_loss : 1.7661, val_loss : 1.7464
epoch : 8, train_loss : 1.7674, val_loss : 1.7644
epoch : 9, train_loss : 1.7418, val_loss : 1.7580
epoch : 10, train_loss : 1.7289, val_loss : 1.7608
accuracy : 0.7095, f1_score: (macro: 0.6957, micro: 0.7095)


=== 인간 1차 예측 ===

[ SYSTEM MESSAGE ]
데이터셋이 battle_dataset/ 경로에 저장되었습니다.
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}

(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)

[ 인간의 1차 하이퍼파라미터 : {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.0, 'lr': 0.000612, 'activation_func': 'leaky_relu', 'optimizer': 'adamw', 'scheduler': 'exp_95'} ]
epoch : 0, train_loss : 2.0474, val_loss : 1.9166
epoch : 1, train_loss : 1.9074, val_loss : 1.8812
epoch : 2, train_loss : 1.8576, val_loss : 1.8531
epoch : 3, train_loss : 1.8394, val_loss : 1.8208
epoch : 4, train_loss : 1.7886, val_loss : 1.7678
epoch : 5, train_loss : 1.7711, val_loss : 1.7712
epoch : 6, train_loss : 1.7468, val_loss : 1.7728
epoch : 7, train_loss : 1.7292, val_loss : 1.7470
epoch : 8, train_loss : 1.7018, val_loss : 1.7675
epoch : 9, train_loss : 1.7079, val_loss : 1.7531
epoch : 10, train_loss : 1.6825, val_loss : 1.7354
epoch : 11, train_loss : 1.6658, val_loss : 1.7378
epoch : 12, train_loss : 1.6550, val_loss : 1.7212
epoch : 13, train_loss : 1.6388, val_loss : 1.7342
epoch : 14, train_loss : 1.6309, val_loss : 1.7326
epoch : 15, train_loss : 1.6291, val_loss : 1.7106
epoch : 16, train_loss : 1.6207, val_loss : 1.7281
epoch : 17, train_loss : 1.6193, val_loss : 1.7071
epoch : 18, train_loss : 1.6140, val_loss : 1.7241
epoch : 19, train_loss : 1.6074, val_loss : 1.7182
epoch : 20, train_loss : 1.6031, val_loss : 1.7190
accuracy : 0.719, f1_score: (macro: 0.706, micro: 0.719)
[ 인간의 Macro F1 Score = 0.706 ]


=== 인간 2차 예측 ===

[ SYSTEM MESSAGE ]
데이터셋이 battle_dataset/ 경로에 저장되었습니다.
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}

(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)

[ 인간의 2차 하이퍼파라미터 : {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.0, 'lr': 0.000612, 'activation_func': 'leaky_relu', 'optimizer': 'adamw', 'scheduler': 'exp_95'} ]
epoch : 0, train_loss : 2.1175, val_loss : 2.0152
epoch : 1, train_loss : 1.9972, val_loss : 1.9640
epoch : 2, train_loss : 1.8750, val_loss : 1.8634
epoch : 3, train_loss : 1.8250, val_loss : 1.8126
epoch : 4, train_loss : 1.7965, val_loss : 1.7886
epoch : 5, train_loss : 1.7841, val_loss : 1.7868
epoch : 6, train_loss : 1.7526, val_loss : 1.7268
epoch : 7, train_loss : 1.7249, val_loss : 1.7291
epoch : 8, train_loss : 1.7095, val_loss : 1.7001
epoch : 9, train_loss : 1.6812, val_loss : 1.6824
epoch : 10, train_loss : 1.6729, val_loss : 1.6879
epoch : 11, train_loss : 1.6597, val_loss : 1.6830
epoch : 12, train_loss : 1.6388, val_loss : 1.6703
epoch : 13, train_loss : 1.6151, val_loss : 1.6791
epoch : 14, train_loss : 1.6060, val_loss : 1.6695
epoch : 15, train_loss : 1.5979, val_loss : 1.6831
accuracy : 0.7524, f1_score: (macro: 0.755, micro: 0.7524)
[ 인간의 Macro F1 Score = 0.755 ]
[ Oh-LoRA 👱‍♀️ 의 Macro F1 Score = 0.6957 ] (예측: 0.6847638487815857)
[ Oh-LoRA 👱‍♀️ 의 하이퍼파라미터 = {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.3, 'dropout_fc': 0.3019303289512014, 'lr': 0.00019712943348718407, 'activation_func': 'leaky_relu', 'optimizer': 'adamw', 'scheduler': 'exp_98'} ]
[ 최종 결과 : 인간 사용자의 승리 ]

 ==== 상세 점수 ====
Human 1st : 0.706
Human 2nd : 0.755
Oh-LoRA   : 0.6957
=================


Oh-LoRA 👱‍♀️ :  처음엔 쪼끔 아슬아슬했는데 두 번째는 완전 잘했어! 😊



[ 데이터셋 이름 : mnist ]
rejected distribution: train=[], test=[]
rejected distribution: train=[1517, 691, 645, 497, 450], test=[234, 127, 113, 98, 91]
rejected distribution: train=[], test=[]
rejected distribution: train=[1275, 560, 523, 392, 357], test=[196, 97, 91, 73, 72]
rejected distribution: train=[233, 69, 45, 44, 31], test=[55, 23, 11, 10, 9]
rejected distribution: train=[2694, 434, 333, 323, 320], test=[449, 60, 44, 36, 31]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[1868, 1802, 1676, 1328, 1319], test=[323, 304, 293, 246, 191]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[3191, 3031, 2905, 2626, 2533], test=[580, 530, 514, 408, 392]
rejected distribution: train=[103, 85, 76, 59, 54], test=[28, 13, 13, 9, 8]
rejected distribution: train=[23, 15, 15, 13, 10], test=[4, 4, 2]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[990, 955, 899, 854, 709], test=[167, 157, 124, 114, 102]
rejected distribution: train=[1103, 1054, 998, 885, 876], test=[214, 168, 153, 153, 136]
rejected distribution: train=[], test=[]
rejected distribution: train=[2406, 2124, 2102, 2055, 1699], test=[399, 354, 343, 330, 282]
rejected distribution: train=[388, 163, 152, 104, 81], test=[47, 28, 21, 19, 17]
rejected distribution: train=[3413, 3075, 2883, 2779, 2505], test=[555, 549, 492, 465, 454]
rejected distribution: train=[5, 1, 1, 1], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[2646, 633, 449, 438, 434], test=[431, 82, 61, 47, 37]
rejected distribution: train=[317, 135, 133, 89, 68], test=[37, 20, 15, 14, 13]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[11, 10, 6, 5, 5], test=[1, 1, 1]
rejected distribution: train=[2157, 294, 233, 208, 200], test=[347, 45, 32, 23, 22]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[1, 1], test=[]
rejected distribution: train=[253, 183, 129, 125, 119], test=[42, 25, 23, 22, 17]
rejected distribution: train=[306, 89, 56, 55, 43], test=[70, 31, 16, 14, 11]
rejected distribution: train=[2788, 2551, 2346, 1899, 1842], test=[474, 455, 437, 320, 271]
rejected distribution: train=[], test=[]
rejected distribution: train=[3812, 2574, 2425, 2183, 1902], test=[578, 458, 399, 396, 362]
rejected distribution: train=[108, 5, 2, 2, 1], test=[23, 2, 1, 1]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[5571, 3259, 2585, 2527, 1966], test=[930, 517, 421, 377, 307]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
rejected distribution: train=[], test=[]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 803/803 [00:01<00:00, 486.56it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 142/142 [00:01<00:00, 102.70it/s] 
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 168/168 [00:01<00:00, 103.75it/s] 


=== Oh-LoRA 👱‍♀️ 선공 예측 ===
epoch : 0, train_loss : 2.1710, val_loss : 1.9774
epoch : 1, train_loss : 1.8227, val_loss : 1.7601
epoch : 2, train_loss : 1.6586, val_loss : 1.5787
epoch : 3, train_loss : 1.5551, val_loss : 1.5300
epoch : 4, train_loss : 1.5331, val_loss : 1.5293
epoch : 5, train_loss : 1.5210, val_loss : 1.5255
epoch : 6, train_loss : 1.5123, val_loss : 1.5219
epoch : 7, train_loss : 1.5086, val_loss : 1.5090
epoch : 8, train_loss : 1.5024, val_loss : 1.5134
epoch : 9, train_loss : 1.4941, val_loss : 1.5078
epoch : 10, train_loss : 1.4936, val_loss : 1.5073
epoch : 11, train_loss : 1.4884, val_loss : 1.5004
epoch : 12, train_loss : 1.4876, val_loss : 1.4998
epoch : 13, train_loss : 1.4859, val_loss : 1.5047
epoch : 14, train_loss : 1.4851, val_loss : 1.4966
epoch : 15, train_loss : 1.4824, val_loss : 1.4958
epoch : 16, train_loss : 1.4816, val_loss : 1.4910
epoch : 17, train_loss : 1.4803, val_loss : 1.4853
epoch : 18, train_loss : 1.4803, val_loss : 1.5009
epoch : 19, train_loss : 1.4789, val_loss : 1.4850
epoch : 20, train_loss : 1.4783, val_loss : 1.4877
accuracy : 0.9762, f1_score: (macro: 0.976, micro: 0.9762)


=== 인간 1차 예측 ===

[ SYSTEM MESSAGE ]
데이터셋이 battle_dataset/ 경로에 저장되었습니다.
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}

(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)

[ 인간의 1차 하이퍼파라미터 : {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.0, 'lr': 0.0003, 'activation_func': 'leaky_relu', 'optimizer': 'adamw', 'scheduler': 'exp_95'} ]
epoch : 0, train_loss : 1.9868, val_loss : 1.6199
epoch : 1, train_loss : 1.5714, val_loss : 1.5326
epoch : 2, train_loss : 1.5188, val_loss : 1.5215
epoch : 3, train_loss : 1.5079, val_loss : 1.5330
epoch : 4, train_loss : 1.4995, val_loss : 1.5076
epoch : 5, train_loss : 1.4913, val_loss : 1.5147
epoch : 6, train_loss : 1.4874, val_loss : 1.5034
epoch : 7, train_loss : 1.4837, val_loss : 1.4944
epoch : 8, train_loss : 1.4807, val_loss : 1.4929
epoch : 9, train_loss : 1.4789, val_loss : 1.4819
epoch : 10, train_loss : 1.4767, val_loss : 1.4833
epoch : 11, train_loss : 1.4764, val_loss : 1.4841
epoch : 12, train_loss : 1.4763, val_loss : 1.4860
accuracy : 0.9702, f1_score: (macro: 0.9679, micro: 0.9702)
[ 인간의 Macro F1 Score = 0.9679 ]


=== 인간 2차 예측 ===

[ SYSTEM MESSAGE ]
데이터셋이 battle_dataset/ 경로에 저장되었습니다.
다음과 같은 형식으로 하이퍼파라미터를 저장하여 battle_dataset/hps.txt 로 저장한 다음 Enter 키를 눌러 주세요.
(이미 battle_dataset/hps.txt 파일이 있다면 최적의 하이퍼파라미터로 수정해 주세요.)

{"dropout_conv_earlier": {0.0 - 0.3 사이의 float 값},
 "dropout_conv_later": {0.0 - 0.3 사이의 float 값},
 "dropout_fc": {0.0 - 0.6 사이의 float 값},
 "lr": {0.00002 - 0.006 사이의 float 값},
 "activation_func": "{relu|leaky_relu}",
 "optimizer": "{adam|adamw}",
 "scheduler": "{exp_80|exp_90|exp_95|exp_98|cosine}"}

(참고: lr은 learning rate 이고, scheduler 중 exp_N 에서 N은 gamma 값 (%) 을 의미합니다.)

[ 인간의 2차 하이퍼파라미터 : {'dropout_conv_earlier': 0.0, 'dropout_conv_later': 0.0, 'dropout_fc': 0.0, 'lr': 2e-05, 'activation_func': 'leaky_relu', 'optimizer': 'adamw', 'scheduler': 'exp_95'} ]
epoch : 0, train_loss : 2.2759, val_loss : 2.2268
epoch : 1, train_loss : 2.1878, val_loss : 2.2031
epoch : 2, train_loss : 2.1694, val_loss : 2.1784
epoch : 3, train_loss : 2.1107, val_loss : 2.0681
epoch : 4, train_loss : 1.9114, val_loss : 1.9020
epoch : 5, train_loss : 1.8284, val_loss : 1.8288
epoch : 6, train_loss : 1.6850, val_loss : 1.6372
epoch : 7, train_loss : 1.5915, val_loss : 1.5987
epoch : 8, train_loss : 1.5699, val_loss : 1.5855
epoch : 9, train_loss : 1.5578, val_loss : 1.5884
epoch : 10, train_loss : 1.5500, val_loss : 1.5810
epoch : 11, train_loss : 1.5401, val_loss : 1.5686
epoch : 12, train_loss : 1.5380, val_loss : 1.5640
epoch : 13, train_loss : 1.5326, val_loss : 1.5636
epoch : 14, train_loss : 1.5297, val_loss : 1.5594
epoch : 15, train_loss : 1.5270, val_loss : 1.5523
epoch : 16, train_loss : 1.5232, val_loss : 1.5494
epoch : 17, train_loss : 1.5213, val_loss : 1.5490
epoch : 18, train_loss : 1.5197, val_loss : 1.5455
epoch : 19, train_loss : 1.5178, val_loss : 1.5466
epoch : 20, train_loss : 1.5171, val_loss : 1.5481
epoch : 21, train_loss : 1.5158, val_loss : 1.5454
accuracy : 0.9702, f1_score: (macro: 0.9695, micro: 0.9702)
[ 인간의 Macro F1 Score = 0.9695 ]
[ Oh-LoRA 👱‍♀️ 의 Macro F1 Score = 0.976 ] (예측: 0.9495638012886047)
[ Oh-LoRA 👱‍♀️ 의 하이퍼파라미터 = {'dropout_conv_earlier': 0.04295631068448605, 'dropout_conv_later': 0.17399977458427057, 'dropout_fc': 0.3481511342335397, 'lr': 0.0001499349925982907, 'activation_func': 'relu', 'optimizer': 'adam', 'scheduler': 'exp_98'} ]

 ==== 상세 점수 ====
Human 1st : 0.9679
Human 2nd : 0.9695
Oh-LoRA   : 0.976
=================


Oh-LoRA 👱‍♀️ :  아쉽다 둘 다 아슬아슬하게 졌네! 다음엔 꼭 이길 수 있을 거야! ✨

(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle> git status
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   battle_dataset/hps.txt
        modified:   ../qa_list.md

no changes added to commit (use "git add" and/or "git commit -a")
(venv) PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2025_10_06_OhLoRA_HP_Battle\hpo_run_battle> 
```

</details>
