
## 목차

* [1. 개요](#1-개요)
* [2. 전체 구조](#2-전체-구조)
* [3. 상세 프로세스](#3-상세-프로세스)
  * [3-1. 코드 리뷰](#3-1-코드-리뷰)
  * [3-2. 사용자 코멘트 도출](#3-2-사용자-코멘트-도출)
  * [3-3. 코드 리뷰 코멘트 생성](#3-3-코드-리뷰-코멘트-생성)

## 1. 개요

* Oh-LoRA Code Assistant 의 전체 코드 리뷰 프로세스

## 2. 전체 구조

![image](../../images/260828_36.PNG)

| 프로세스         | 입력                        | 출력                                                             | 처리 과정                                                                                                       |
|--------------|---------------------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 코드 리뷰        | Python 코드베이스              | 각 rule 별 위반 부분 검출 결과                                           | - Rule-based check (45개 규칙, 13개의 `ruff` 규칙 포함)<br>- 🤖 임베딩 기반 check (11개 규칙)                                |
| 사용자 코멘트 도출   | 각 rule 별 위반 부분 검출 결과      | - 각 rule 별 상세 코멘트<br>- 점수가 가장 낮은 5개의 rule 관련 코멘트<br>- 전체 평균 점수 | - 검출 결과로부터 상세 코멘트 생성<br>- 각 rule 별 **준수 정도를 나타내는 점수** 를 일정한 수식에 의해 계산<br>- 전체 평균 점수는 **각 rule 별 점수의 산술 평균** |
| 코드 리뷰 코멘트 생성 | 점수가 가장 낮은 5개의 rule 관련 코멘트 | 코드 리뷰 코멘트                                                      | 🤖 LLM은 `kanana-2-3b-instruct` 모델 사용                                                                        |

## 3. 상세 프로세스

### 3-1. 코드 리뷰

Python Codebase 를 **Rule-based 기반 검사 규칙** 및 **임베딩 기반 검사 규칙** 에 의해 검사하고, 각 rule 별 위반된 부분을 검출한다.

* [코드 리뷰 기준 (전체 rule)](code_review_standard.md)
* Rule-based check (45개 규칙)
  * [ruff 의 한계점 분석 결과 (Oh-LoRA 자체 규칙 32개의 도입 이유)](ruff_limits.md)
* Embedding-based check (11개 규칙)
  * [각 규칙 별 선택한 임베딩 모델](../embedding_model/model_candidates.md)
  * [각 규칙 별 임베딩 모델 테스트 결과](../embedding_model/model_test_result.md)

### 3-2. 사용자 코멘트 도출

**각 rule 별 위반 부분 검출 결과** 에 따라 **각 rule 별 상세 코멘트, 점수가 가장 낮은 5개의 rule 관련 코멘트, 전체 평균 점수** 도출

* 각 rule 별 상세 코멘트
  * **규칙 별 > 파일 별 > 함수 별** 로 아래 예시와 같이 상세 코멘트 도출

```
import 순서 준수:
파일: test_cases\code_review_items_old.py
 - 함수: (최상위 레벨)
   - line 6 에 있는 import of builtin "re"
   - line 7 에 있는 import of builtin "tokenize"
   - line 8 에 있는 import of builtin "keyword"
   - line 9 에 있는 import of builtin "builtins"
   - line 11 에 있는 import of builtin "typing"
   - line 12 에 있는 import of builtin "difflib"
   - line 13 에 있는 import of builtin "operator"
   - line 17 에 있는 import of builtin "collections"
   - line 18 에 있는 import of builtin "itertools"

파일: test_cases\tc01_01_basics.py
 - 함수: (최상위 레벨)
   - line 5 에 있는 import of builtin "os"

파일: test_cases\tc01_08.py
 - 함수: (최상위 레벨)
   - line 6 에 있는 import of builtin "sys"
   - line 7 에 있는 import of builtin "collections"
   - line 10 에 있는 import of builtin "re"
   - line 11 에 있는 import of 3rd-party "torch"


함수 docstring, 함수명 의미 일치:
파일: test_cases\ai_test.py
 - 함수: throttle
   - line 32 에 있는 함수 throttle - docstring 불일치

 - 함수: test_1
   - line 40 에 있는 함수 test_1 단일 책임 원칙 위반
   - line 40 에 있는 함수 test_1 - docstring 불일치

 - 함수: test_2
   - line 45 에 있는 함수 test_2 - docstring 불일치

 - 함수: test_3
   - line 49 에 있는 함수 test_3 단일 책임 원칙 위반
   - line 49 에 있는 함수 test_3 - docstring 불일치

 - 함수: test_4
   - line 53 에 있는 함수 test_4 - docstring 불일치

 - 함수: test_5
   - line 58 에 있는 함수 test_5 - docstring 불일치
```

* 각 rule 별 준수 정도 점수 계산
  * 각 파일 별, **`해당 파일의 line 수` - 100 x `위반 건수`** 를 계산 (최솟값 0)
  * 해당 값을 `모든 파일의 line 수 합계` 로 나눈 값을 계산
  * 이 값을 100점으로 환산

* 전체 평균 점수
  * 모든 rule 별, 위 기준으로 계산한 점수의 산술 평균 

### 3-3. 코드 리뷰 코멘트 생성

**점수가 가장 낮은 5개의 rule** 정보를 기반으로, LLM (Kanana-2 3B) 을 이용한 코드 리뷰 코멘트 생성

* [LLM 선택 근거 및 테스트 결과 (ChatGPT-as-a-judge)](../llm_model/llm_selection.md)
