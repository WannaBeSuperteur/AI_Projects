
* EffiSegNet Base Implementation from [Official PyTorch Implementation](https://github.com/ivezakis/effisegnet/tree/main)

## TEST RESULT

* with **EffiSegNet-B4 (Pre-trained)**
  * **50 epochs** 로 테스트 시, Original Paper 와 전반적으로 **큰 차이 없음 (테스트 결과 정상 판단)** 

| 구분                                    | F1 Score | Dice Score | IoU Score | Precision | Recall | train log<br>(csv file)                                                                                                                                                |
|---------------------------------------|----------|------------|-----------|-----------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original Paper                        | 0.9552   | 0.9483     | 0.9056    | 0.9679    | 0.9429 |                                                                                                                                                                        |
| test (50 epochs)                      | 0.9506   | 0.9445     | 0.8980    | 0.9647    | 0.9368 | [train log](https://github.com/WannaBeSuperteur/AI_Projects/blob/7261c19b584666b1a9c7fec07888804fbca3d832/2025_05_22_Improve_EffiSegNet/effisegnet_base/train_log.csv) |
| test (300 epochs = Original Paper 조건) | 0.9488   | 0.9413     | 0.8965    | 0.9671    | 0.9311 | [train log](https://github.com/WannaBeSuperteur/AI_Projects/blob/bc0cb10c77e988f727856444f9b2a91b9f5ce803/2025_05_22_Improve_EffiSegNet/effisegnet_base/train_log.csv) |