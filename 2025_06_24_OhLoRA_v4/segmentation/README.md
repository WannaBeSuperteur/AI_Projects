## 코드 실행 방법

모든 코드는 ```2025_06_24_OhLoRA_v4``` (프로젝트 메인 디렉토리) 에서 실행 **(단, 먼저 HuggingFace Link (TBU) 에서 모델 파일 다운로드 후, 모델 파일 경로 정보 (TBU) 에 따라 해당 파일들을 알맞은 경로에 배치)**

* **FaceXFormer** 모델 실행
  * Segmentation 대상 : [Oh-LoRA 👱‍♀️ StyleGAN-FineTune-v8 학습 데이터 4,930 장](../../2025_05_26_OhLoRA_v3/stylegan/README.md#1-1-모델-구조)
  * ```python segmentation/run_segmentation.py```

* Oh-LoRA v4 용 경량화된 Segmentation 모델 실행
  * Segmentation 대상 : [Oh-LoRA 👱‍♀️ StyleGAN-FineTune-v8 학습 데이터 4,930 장](../../2025_05_26_OhLoRA_v3/stylegan/README.md#1-1-모델-구조)
  * ```python segmentation/run_seg_model_ohlora_v4.py```

## 성능 테스트 결과

* [해당 문서](test_result.md) 참고.
