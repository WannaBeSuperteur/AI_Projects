## 1. 개요

* **Oh-LoRA 👱‍♀️ (오로라) 프로젝트의 v3 버전** 에서 사용하는 **Property Score CNN (핵심 속성 값 계산용)**
* 핵심 속성 값 정보 [(상세 설명은 오로라 v1 개발 문서 참고)](../../2025_04_08_OhLoRA/stylegan_and_segmentation/README.md#2-핵심-속성-값)

| 핵심 속성 값                                  | 설명                                               | 용도                                                                                                   |
|------------------------------------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 눈을 뜬 정도 ```eyes```                       | Oh-LoRA 👱‍♀️ 가 **눈을 크게 뜰수록** 큰 값                | Oh-LoRA 👱‍♀️ 의 표정/몸짓 제어 & 표시                                                                        |
| 입을 벌린 정도 ```mouth```                     | Oh-LoRA 👱‍♀️ 가 **입을 크게 벌릴수록** 큰 값               | 상동                                                                                                   |
| 고개 돌림 정도 ```pose```                      | Oh-LoRA 👱‍♀️ 가 **고개를 오른쪽으로 많이 돌릴수록** 큰 값        | 상동                                                                                                   |
| 머리 색 ```hair_color```                    | Oh-LoRA 👱‍♀️ 의 **머리 색이 밝을수록** 큰 값               | StyleGAN-VectorFind-v8 의 ```eyes``` ```mouth``` ```pose``` 값을 가장 잘 변화시키는 벡터를 SVM 으로 찾을 때, 이미지 그룹화 기준 |
| 머리 길이 ```hair_length```                  | Oh-LoRA 👱‍♀️ 의 **머리가 길수록** 큰 값                  | 상동                                                                                                   |
| 배경 색 밝기 (각 픽셀의 평균) ```background_mean``` | Oh-LoRA 👱‍♀️ 가 있는 **배경이 전체적으로 밝을수록** 큰 값        | 상동                                                                                                   |
| 직모 or 곱슬머리 ```hair_style```              | Oh-LoRA 👱‍♀️ 의 헤어스타일이 **직모보다는 곱슬머리에 가까울수록** 큰 값 | 상동 **(Oh-LoRA v3 에서 추가)**                                                                            |

## 코드 실행 방법

모든 코드는 ```2025_05_26_OhLoRA_v3``` (프로젝트 메인 디렉토리) 에서 실행

* **StyleGAN-FineTune-v8** 의 Fine-Tuning 데이터셋에 대한 Face Segmentation 결과 생성
  * ```python property_score_cnn/run_segmentation.py``` 

* hairstyle score 계산
  * ```python property_score_cnn/run_compute_hairstyle_score.py``` 