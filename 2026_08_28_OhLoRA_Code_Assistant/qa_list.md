# Oh-LoRA Code Assistant QA List

## 목차

* [1. 기본 사항](#1-기본-사항)
* [2. QA 리스트](#2-QA-리스트)
  * [2-1. 테스트 케이스 목록](#2-1-테스트-케이스-목록) 

## 1. 기본 사항

* 본 문서는 **Oh-LoRA 👱‍♀️ (오로라) Code Assistant** 에 대한 체계적인 검증을 위한 QA List 이다.
* 1차 검사 통과 시, **1차 검사 결과** 를 최종 결과로 하고 **최종 검사 결과** 란은 비운다.
* 검사 결과는 이모지 (✅ or ❌) 로 나타낸다.

## 2. QA 리스트

| 번호 | 검사 항목                                                                                                                                                  | 1차 검사 결과 | 버그 수정 | 최종 검사 결과 |
|----|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-------|----------|
| 1  | 모델 다운로드 정상 동작 (사용자 PC 기준)                                                                                                                              |          |       |          |
| 2  | API 실행 시 정상 동작 (서버 PC 기준)                                                                                                                              |          |       |          |
| 3  | [사용자 프로그램](user_program/main.py) 테스트 케이스 실행 결과 일치                                                                                                      |          |       |          |
| 4  | [code_reviewer.py](code_reviewer/code_reviewer.py) 실행 시 임베딩 로그가 [기존 모델 로그](code_reviewer/bugfix_logs/ai_code_check_log_before_model_download.csv) 와 동일 |          |       |          |

### 2-1. 테스트 케이스 목록

* 전체 테스트 케이스는 [사용자 프로그램 코드 파일](user_program/main.py) 의 docstring 참고.

| 번호  | 테스트 목적                       | 테스트 케이스 (Windows shell 명령어)                                                                                                                                                      | 1차 검사 결과 | 버그 수정 | 최종 검사 결과 |
|-----|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-------|----------|
| 3-1 | 기본 형태 (코드 경로)                | `python main.py --code_path ../code_reviewer/test_cases --verbose True`                                                                                                          |          |       |          |
| 3-2 | `except_path` (제외 경로) 포함     | `python main.py --code_path ../code_reviewer/test_cases --except_path ../code_reviewer/test_cases/path_test --verbose True`                                                      |          |       |          |
| 3-3 | 단일 파일 테스트                    | `python main.py --code_path ../code_reviewer/test_cases/tc02.py --verbose True`                                                                                                  |          |       |          |
| 3-4 | 다중 파일 테스트                    | `python main.py --code_path "../code_reviewer/test_cases/tc02.py \|\|\| ../code_reviewer/test_cases/tc03.py" --verbose True`                                                     |          |       |          |
| 3-5 | 단일 파일 테스트 (`except_path` 포함) | `python main.py --code_path ../code_reviewer/test_cases/tc02.py --except_path ../code_reviewer/test_cases/path_test --verbose True`                                              |          |       |          |
| 3-6 | 다중 파일 테스트 (`except_path` 포함) | `python main.py --code_path "../code_reviewer/test_cases/tc02.py \|\|\| ../code_reviewer/test_cases/tc03.py" --except_path ../code_reviewer/test_cases/path_test --verbose True` |          |       |          |

