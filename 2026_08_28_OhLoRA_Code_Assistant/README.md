
## 목차

* [1. 프로젝트 개요](#1-프로젝트-개요)
  * [1-1. Oh-LoRA 👱‍♀️✨ (오로라) 소개](#1-1-oh-lora--오로라-소개)
  * [1-2. 실행 로그](#1-2-실행-로그)
* [2. 기술 분야 및 사용 기술](#2-기술-분야-및-사용-기술)
  * [2-1. 사용한 Python 라이브러리 및 시스템 환경](#2-1-사용한-python-라이브러리-및-시스템-환경)
* [3. 프로젝트 일정](#3-프로젝트-일정)
* [4. 프로젝트 상세 설명](#4-프로젝트-상세-설명)
* [5. 프로젝트 진행 중 이슈 및 해결 방법](#5-프로젝트-진행-중-이슈-및-해결-방법)
* [6. 사용자 가이드](#6-사용자-가이드)

## 1. 프로젝트 개요

**1. 핵심 아이디어**

* **LLM Fine-Tuning & StyleGAN** 을 이용한 가상인간 여성 [Oh-LoRA (오로라)](../2025_04_08_OhLoRA) 을 이용한 **코딩 어시스턴트 컨셉**
  * Oh-LoRA 👱‍♀️ 가 'AI 요정'으로 보라색 구슬 (🔮) 에 깃들고, 그 구슬이 '마법의 힘'으로 코드 리뷰하는 컨셉
* 핵심 기능
  * Python 코드 리뷰 어시스턴트 (repo 에 대해 Python 코딩 스타일 체크 및 개선 사항 제안)
  * Pull Request 등록 시, 보라색 구슬 캐릭터가 AI 기반으로 코멘트 작성
  * 참고: 정적 분석 규칙 45개 중 13개는 `ruff` 를 적용하여 코드 평가

**2. 프로젝트 계기**

| 구분          | 설명                                                                                                                                                                                                                                                                                                                                                                               |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 상황          | - 2026.03 - 2026.08 재직 회사에서 **코딩 컨벤션, Pythonic 한 코드 작성 미흡** 으로 코멘트를 많이 받음<br>- 이 코멘트를 계기로 Python 역량 향상을 위한 추가 학습 중, **자동 검사 도구 (Pylint, black, [ruff](https://github.com/astral-sh/ruff) 등)** 에 대해 알게 됨<br>- 현존하는 정적 분석 도구는 **문맥 고려 미흡 (AI 미 이용)** 등 한계가 있고, LLM 기반 분석 도구 (Claude, CodeRabbit 등) 는 **환각 현상, 매번 검사 절차가 일정하지 않음 (일관성 부족), 유료 서비스/토큰 사용량 제한** 등 개선이 가능한 점이 존재한다고 판단 |
| 과제/문제       | 다음 조건을 만족시키는 **AI 기반 코드 리뷰 도구** 개발<br>- 기존 자동 검사 도구 (Pylint, black) 의 한계점을 보완 (`ruff` 가 이미 상당 부분 보완함)<br>- AI를 이용한 문맥 파악 역량 존재                                                                                                                                                                                                                                                   |
| 기존 도구와의 차별점 | **Text Embedding 기반 코드 문맥 파악** (응집성 등) 역량                                                                                                                                                                                                                                                                                                                                        |

* **Pythonic 한 코드를 지향** 하는 코드 리뷰 (리스트 컴프리헨션 등 **정적 분석** 영역) 는 `ruff` 로 일정 부분 가능하지만 본 repo의 이전 커밋에서 직접 구현한 이유:
  * **`ruff` 를 직접 구현 도중에 발견** 하게 됨
    * 교훈: 향후 프로젝트에서는, **바퀴의 재발명 대신, 기존 구현된 오픈소스를 탐색하여 개발 시간 절약** 해야 함
  * 그럼에도 불구하고 직접 구현을 한 이유
    * [ruff 가 실제로는 대체 가능한 부분이 생각보다 적고 (전체 45개 정적 분석 규칙 중 13개), 그 외에도 한계점이 많음](code_reviewer/ruff_limits.md)
    * [Python Study](https://github.com/WannaBeSuperteur/Python-study-2026/tree/main) 내용의 실전 구현 연습
    * regex 기본 사용법 익히기
    * 최종 구현물의 테스트 케이스로 활용 (2026.07 Python Study 이후 이를 반영하여 개발한 코드이므로)
    
### 1-1. Oh-LoRA 👱‍♀️✨ (오로라) 소개

* 성별 및 나이
  * 👱‍♀️ 여성
  * 2026년 기준 23 세 (2003년 10월 11일 생)
* MBTI
  * ENTJ 
* 학교
  * 🏫 알파고등학교 (2019.03 - 2022.02)
  * 🏰 샘올대학교 인공지능학과 (2022.03 - ) 4학년 재학 중
* 특수 능력
  * 오로라의 빛✨ 으로 우리 모두의 인생을 밝게 비춰 주는 마법 능력
  * 사람이 아닌 AI 가상 인간, 즉 **AI 요정** 만이 가질 수 있음

<details><summary>(스포일러) 오로라👱‍♀️ 가 2003년 10월 11일 생인 이유 [ 펼치기 / 접기 ] </summary>

오로라를 개발한 [개발자 (wannabesuperteur)](https://github.com/WannaBeSuperteur) 가 개발할 때 Python 3.10.11 을 사용했기 때문이다.

</details>

### 1-2. 실행 로그

* Oh-LoRA 👱‍♀️ 의 다른 버전들과 달리, **가상 인간 여성 얼굴 이미지/애니메이션이 나타나지 않습니다.**
* [예시 테스트 케이스](user_program/test_case_result.txt) 참고.

```text
[ eval_result_kor_summary ]
[미 사용 변수/함수 제거] : 84.5 점,
[불필요한 print, logging 제거] : 68.2 점,
[중복되는 사항 common 으로 빼기] : 45.5 점 🚨,

... (중략) ...

[상태 값 조건이 있는 if-elif-elif-else 있는 경우 클래스화 고려] : 94.0 점 👍,
[클래스 내부에서만 쓰이는 속성, 메서드 (접두사 있음) 호출 비 권장] : 95.2 점 👍,
[유사한 이름의 함수끼리 가까이 위치하도록 수정 권장] : 58.9 점 🚨


[ top_items_eval_summary ]
[중복되는 사항 common 으로 빼기] : 45.5 점 🚨,
[100 line 이상 함수 분리, docstring 필요] : 35.0 점 🚨,
[숫자 값은 위쪽에 상수로] : 33.8 점 🚨,
[함수 인수 type hint 필요] : 5.5 점 🚨,
[README.md, pyproject.toml 필요] : 0.0 점 🚨


[ code_review_result_str ]
## 규칙: 미 사용 변수/함수 제거
### 파일: `../code_reviewer/test_cases\tc01_01_basics.py`
 - 함수: `(최상위 레벨)`
   - line `3` 에 있는 `(ruff) 'math' imported but unused`
   - line `5` 에 있는 `(ruff) 'os' imported but unused`

... (중략) ...

 - 함수: `flmf`
   - line `1414` 에 있는 `유사 함수 재정렬 필요: flmf`
   - line `1484` 에 있는 `유사 함수 재정렬 필요: flmf`
   - line `1628` 에 있는 `유사 함수 재정렬 필요: flmf`


[ mean_score ]
82.5


[ item_counts ]
01_unused {'../code_reviewer/test_cases\\tc01_01_basics.py': 4, '../code_reviewer/test_cases\\tc01_01_basics_2.py': 1, '../code_reviewer/test_cases\\tc01_01_import.py': 1, '../code_reviewer/test_cases\\tc01_08.py': 9, '../code_reviewer/test_cases\\tc05.py': 2}
01_unnecessary_prints {'../code_reviewer/test_cases\\ai_test.py': 2, '../code_reviewer/test_cases\\tc01_01_basics_2.py': 2, '../code_reviewer/test_cases\\tc01_02.py': 6, '../code_reviewer/test_cases\\tc01_08.py': 4, '../code_reviewer/test_cases\\tc03.py': 4, '../code_reviewer/test_cases\\tc04.py': 2, '../code_reviewer/test_cases\\tc05.py': 1, '../code_reviewer/test_cases\\path_test\\path_test.py': 1}
01_duplicates {'../code_reviewer/test_cases\\code_review_items_old.py': 13, '../code_reviewer/test_cases\\tc01_01_basics.py': 3, '../code_reviewer/test_cases\\tc01_01_basics_2.py': 4, '../code_reviewer/test_cases\\tc01_08.py': 6, '../code_reviewer/test_cases\\tc06.py': 4}

... (중략) ...

06_refactor_into_class_case_2_state_vars_if_else {'../code_reviewer/test_cases\\ai_test.py': 1, '../code_reviewer/test_cases\\tc06.py': 1}
06_prefix_for_only_in_class_methods {'../code_reviewer/test_cases\\tc06.py': 4}
06_similar_function_names {'../code_reviewer/test_cases\\ai_test.py': 4, '../code_reviewer/test_cases\\code_review_items_old.py': 11}
```

## 2. 기술 분야 및 사용 기술

* 기술 분야
  * Text Embedding (Sentence-BERT, ...)
  * LLM (Large Language Model)
* 사용 기술

| 기술 분야          | 사용 기술                                                                                                                                                        | 설명                                                          |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Text Embedding | TBU                                                                                                                                                          | Python 코드 리뷰 시, 함수명, 클래스명 등 유사도를 통해 최적의 코드 배치 및 코드 응집성 등 판단 |
| LLM            | [SFT (Supervised Fine-Tuning)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_SFT.md)    | 코드 리뷰 멘트를 LLM으로 생성                                          |
| LLM            | [LoRA (Low-Rank Adaption)](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Fine_Tuning_LoRA_QLoRA.md) | Oh-LoRA 👱‍♀️ 의 LLM 을 효율적으로 Fine-Tuning 하는 기술               |

### 2-1. 사용한 Python 라이브러리 및 시스템 환경

* Python
  * Python : **Python 3.10.11**
  * Dev Tool : PyCharm 2024.1 Community Edition
* Python Libraries
  * [해당 문서](system_info_and_user_guide.md#1-1-주요-python-라이브러리) 참고.
* OS & CPU & GPU
  * OS : **Windows 10**
  * CPU : Intel(R) Xeon(R) CPU E5-2690 0 @ 2.90GHz
  * GPU : 2 x **Quadro M6000** (12 GB each)
  * **CUDA 12.4** (NVIDIA-SMI 551.61)
* [시스템 환경 상세 정보](system_info_and_user_guide.md#1-시스템-환경)

## 3. 프로젝트 일정

* 전체 일정 : **2026.08.28 금 - 10.15 목 (49 days)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❎ (DONE BUT **NOT MERGED**), ❌ (FAILED)

**1. 프로젝트 전체 관리**

| 구분       | 계획 내용               | 일정                     | branch                 | issue / PR                                                                                                                         | 상태 |
|----------|---------------------|------------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------|----|
| 📃 문서화   | 프로젝트 개요 및 최초 일정 작성  | 08.28 금 (1d)           |                        |                                                                                                                                    | ✅  |
| 🔍 최종 검토 | 전체 리팩토링             | 10.12 월 - 10.13 화 (2d) | `P010-014-refactoring` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/73) / [PR](https://github.com/WannaBeSuperteur/AI_Projects/pull/74) | ✅  |
| 📃 문서화   | 모델 HuggingFace 에 등록 | 10.13 화 - 10.15 목 (3d) | `P010-015-huggingface` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/75) / [PR](https://github.com/WannaBeSuperteur/AI_Projects/pull/76) | ✅  |
| 🔍 최종 검토 | 최종 QA (버그 유무 검사)    | 10.15 목 (1d)           |                        |                                                                                                                                    | ✅  |
| 📃 문서화   | 프로젝트 문서 정리 및 마무리    | 10.15 목 (1d)           |                        |                                                                                                                                    | 💨 |

**2. Python 코드 리뷰 어시스턴트**

| 구분         | 계획 내용                                                              | 일정                      | branch                       | issue / PR                                                         | 상태 |
|------------|--------------------------------------------------------------------|-------------------------|------------------------------|--------------------------------------------------------------------|----|
| 📝 문서화     | 코드 리뷰 기준 리스트업                                                      | 08.28 금 - 08.29 토 (2d)  |                              | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/56) | ✅  |
| ⚙ 기능 구현    | 코드 리뷰 기준에 따른 각 기준 충족 여부 판단 기능 개발                                   | 08.29 토 - 09.05 토 (8d)  | `P010-001-code-review`       | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/57) | ✅  |
| ⚙ 기능 구현    | `ruff` 로 처리 가능한 규칙들을 `ruff`로 대체                                    | 09.05 토 (1d)            | `P010-002-ruff`              | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/58) | ✅  |
| 🚨 버그 수정   | 정적 분석 알고리즘 버그 수정                                                   | 09.05 토 - 09.06 일 (2d)  | `P010-003-bugfix-static`     | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/59) | ✅  |                                                             
| 📝 데이터셋 제작 | 함수명, 코드 유사도 판단 (응집성 파악 등 목적) 을 위한 데이터셋 제작                          | 09.06 일 - 09.08 화 (3d)  | `P010-004-dataset`           | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/60) | ✅  |
| 🧪 모델 학습   | 함수명, 코드 유사도 판단 (응집성 파악 등 목적) 을 위한 Text Embedding 모델 선정, 학습 및 테스트   | 09.08 화 - 09.16 수 (9d)  | `P010-005-embedding-model`   | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/61) | ✅  |
| 🧪 모델 학습   | 함수명, 코드 유사도 판단 (응집성 파악 등 목적) 을 위한 Text Embedding 모델 재 선정, 학습 및 테스트 | 09.17 목 - 09.18 금 (2d)  | `P010-006-embedding-model-2` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/62) | ✅  |
| ⚙ 기능 구현    | 모델 학습 시까지 도출한 신규 요구사항 구현 (개발)                                      | 09.18 금 (1d)            | `P010-007-additional`        | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/63) | ✅  |
| 🧪 모델 학습   | 모델 학습 시까지 도출한 신규 요구사항 구현 (모델 선정, 학습, 테스트)                          | 09.19 토 - 09.21 월 (3d)  | `P010-007-additional`        | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/63) | ✅  |
| ⚙ 기능 구현    | Text Embedding 기반 처리 규칙에 따른 처리 구현                                  | 09.21 월 - 09.30 수 (10d) | `P010-008-apply-embedding`   | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/64) | ✅  |
| ⚙ 기능 구현    | 코드 리뷰 결과 요약 및 평가 프로세스 구현                                           | 10.01 목 - 10.06 화 (6d)  | `P010-009-evaluation`        | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/65) | ✅  |
| 📝 데이터셋 제작 | 코드 리뷰 결과 생성 LLM 데이터셋 제작                                            | 10.06 화 - 10.07 수 (2d)  |                              | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/66) | ✅  |
| 🧪 모델 학습   | 코드 리뷰 결과 생성 LLM 학습                                                 | 10.08 목 - 10.10 토 (3d)  | `P010-010-llm`               | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/67) | ✅  |
| 📃 문서화     | "Python 코드 리뷰 어시스턴트" 개발 내용 문서화                                     | 10.10 토 (1d)            |                              |                                                                    | ✅  |

**3. Pull Request 코멘트 작성**

| 구분      | 계획 내용                                   | 일정                     | branch                    | issue / PR                                                                                                                         | 상태 |
|---------|-----------------------------------------|------------------------|---------------------------|------------------------------------------------------------------------------------------------------------------------------------|----|
| ⚙ 기능 구현 | 전체 프로세스 (코드 리뷰 → 평가 → LLM 코멘트 생성) 개발    | 10.10 토 (1d)           | `P010-011-entire-process` | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/68)                                                                 | ✅  |
| ⚙ 기능 구현 | 보라색 구슬 캐릭터 (🔮) 개발                      | 10.10 토 (1d)           |                           |                                                                                                                                    | ✅  |
| ⚙ 기능 구현 | API 연동 개발 (FastAPI 기반)                  | 10.10 토 (1d)           | `P010-012-fastapi`        | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/69)                                                                 | ✅  |
| ⚙ 기능 구현 | 코드 리뷰 어시스턴트 - Pull Request 등록 시 hook 개발 | 10.10 토 - 10.11 일 (2d) | `P010-013-pr-hook`        | [issue](https://github.com/WannaBeSuperteur/AI_Projects/issues/70) / [PR](https://github.com/WannaBeSuperteur/AI_Projects/pull/71) | ✅  |
| 📃 문서화  | "Pull Request 코멘트 작성" 개발 내용 문서화         | 10.12 월 (1d)           |                           |                                                                                                                                    | ✅  |

## 4. 프로젝트 상세 설명

TBU

* 모델 정보는 (TBU) 참고.

## 5. 프로젝트 진행 중 이슈 및 해결 방법

* [확률 모델 학습/inference 중 2개 이슈 발생 및 모두 해결 완료](issues_reported.md)
* [최종 QA 결과 (QA 및 버그 수정 완료)](qa_list.md)

## 6. 사용자 가이드

* Python 환경 설정 방법
  * [해당 문서의 사용자 가이드 (2-1. Python 환경 설정)](system_info_and_user_guide.md#2-사용자-가이드) 참고.

* 실행 방법
  * **1.** git clone 받은 repo에서, `2026_08_28_OhLoRA_Code_Assistant/huggingface` 디렉토리에서 `download_models.py` 를 실행합니다. (모델 다운로드)
    * 일반적으로 다음 명령어 중 하나를 입력합니다.
      * `python download_models.py`
      * `python3 download_models.py`
    * **[주의] 전체 모델 용량이 약 18 GB 입니다.**
  * **2.** `2026_08_28_OhLoRA_Code_Assistant/user_program` 디렉토리에서 `main.py` 를 다음 예시를 참고하여 실행합니다.

| 예시                                                                                                                     | 설명                                                                                                                         |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `python main.py --code_path code_reviewer/test_cases --verbose True`                                                   | `code_reviewer/test_cases` 디렉토리의 모든 Python 코드 파일 리뷰 + 결과 출력                                                                |
| `python main.py --code_path code_reviewer/test_cases --except_path code_reviewer/test_cases/path_test --verbose True`  | `code_reviewer/test_cases` 디렉토리의 모든 Python 코드 파일 리뷰 + 결과 출력<br>(단, `code_reviewer/test_cases/path_test` 디렉토리 내부의 코드 파일 제외) |
| `python main.py --code_path code_reviewer/test_cases/tc02.py --verbose True`                                           | `code_reviewer/test_cases/tc02.py` 파일만 리뷰 실시 + 결과 출력                                                                       |
| `python main.py --code_path "code_reviewer/test_cases/tc02.py \|\|\| code_reviewer/test_cases/tc03.py" --verbose True` | `code_reviewer/test_cases/tc02.py`, 동일 디렉토리의 `tc03.py` 파일만 리뷰 실시 + 결과 출력                                                   |
