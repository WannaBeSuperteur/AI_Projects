# 코드 리뷰 기준

🔮 표시는 **Text Embedding (AI)** 사용이 필요한 부분.

## 목차

* [1. Python 관련](#1-python-관련)
  * [1-1. 기본 사항](#1-1-기본-사항)
  * [1-2. 기본 코딩 컨벤션](#1-2-기본-코딩-컨벤션)
  * [1-3. Python 문법 간소화](#1-3-python-문법-간소화)
  * [1-4. 그 외 Pythonic 한 프로그래밍](#1-4-그-외-pythonic-한-프로그래밍)
  * [1-5. 예외 및 오류 처리](#1-5-예외-및-오류-처리)
  * [1-6. 코드 응집성, 클래스화 등](#1-6-코드-응집성-클래스화-등)
* [2. PyTorch 관련](#2-pytorch-관련)

## 1. Python 관련

### 1-1. 기본 사항

* 미 사용은 없어야 함
  * 미 사용 변수, 함수 등이 없어야 함
  * 미 사용 import 가 없어야 함
* 🔮 불필요한 print, logging 등이 없어야 함
* 중복되는 사항들은 common 으로 빼는 것이 좋음
  * 공통으로 사용되는 상수
  * 기능이나 내부 코드 구조가 거의 유사한 함수
  * 2~3번 이상 반복되는 특정 string
* 🔮 유사한 변수명은 하나로 통일 시키는 것이 좋음
  * 예를 들어 `model_id` `modelid` 뿐만 아니라, `customer_name` `consumer_name` 역시 통일 필요
  * `aaa` `aaa_` 또는 `aaa` `aaas`와 같은 패턴은 서로 구분할 필요 있음
* 🔮 동일/유사한 인자들의 type은 모든 함수에서 하나로 통일
* 🔮 변수명, 함수명은 `a = ...` 등이 아닌, 의미가 있어야 함 (+ 알기 쉽게 할것)
* 🔮 함수명과 반환값이 서로 잘 match 되어야 함 (함수 반환 부분에서 판단)
* `import` 순서 (표준 라이브러리 > 서드파티 라이브러리 > 로컬) 준수 여부
* 🔮 함수의 단일 책임 원칙 (실제로는 `docstring`을 이용하여 판단)
* 🔮 함수 docstring과 함수명이 서로 일치하는지 (혼동을 야기하지 않는지) 판단
* 주석 처리된 코드 제거
* 빈 파일이 있는 경우 `TODO` 로 명시 필요

### 1-2. 기본 코딩 컨벤션

* 🔮 고정값은 맨 위쪽에 상수로 빼 놓는 것이 좋음
* 한 줄의 길이는 120 글자 이내여야 함
* `pyproject.toml` 을 이용한 프로젝트 관리 필요
* `README.md` 가 있으면 좋음
* 함수 관련
  * 함수 길이가 100 줄 이상인 경우 단일 책임 원칙으로 분리 권장
  * 함수 설명에 `docstring` 권장
  * type hinting 필수
* 반복문 중첩으로 인해 들여쓰기가 지나치게 많이 된 경우 별도 함수로 분리, 리스트 컴프리헨션 등 권장

### 1-3. Python 문법 간소화

* 리스트 컴프리헨션 사용이 가능한 경우, 리스트 컴프리헨션 사용
  * 조건 분기와 함께 적용 시 컴프리헨션 적용 가능한 경우
  * `join` 과 함께 리스트 컴프리헨션 적용 가능한 경우 등 포함
  * 단순한 변환이 아닌 경우, 함수형 도구 (`map`, `filter`, `reduce`) 대신 리스트 컴프리헨션 사용 권장
* 제너레이터 표현식 사용이 가능한 경우, 리스트 컴프리헨션 대신 되도록 제너레이터 표현식 사용
* `if-elif-else` 가 반복되는 경우 `dict` 를 이용하여 단순화 (조건 분기 남발 자제)
* 경로를 `a/b/c` 형태가 아닌 `pathlib` 또는 `os.path.join` 사용
* 불필요한 변수 생성 자제 (메모리 절약 목적)
  * 불필요한 변수 생성 대신 컴프리헨션 또는 `defaultdict` 등 사용 권장
* 조건문 중첩 대신 `any`, `all` 사용
* `zip` 사용 가능한 경우 사용
* `enumerate` 사용 가능한 경우 사용
* 아래에서 `before`로 된 부분은 `pathlib`을 사용하여 `after`처럼 사용해야 함

```python
# before
with open('example.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# after
content = Path('example.txt').read_text(encoding='utf-8')
```

* 아래와 같은 형태는 간결하게 바꾸도록 함 (`if xxx`, `if not xxx`)

```python
if len(sentence):
if not len(sentence):
if len(sentence) == 0:
if not len(sentence) == 0:
if sentence == '':
if not sentence == '':
```

* `a['b']`가 None이면 `a['c']`를 사용한다는 문법은 아래와 같이 간소화

```python
result = a['b'] or a['c']
```

* 기존 배열의 원소를 하나씩 추가하는 대신 `extend` 함수 사용
* 개수 세기에 `count` 함수 사용
* 인덱스 반환에 `index` 함수 사용
* str 단순 `+=` 대신 `join` 사용

### 1-4. 그 외 Pythonic 한 프로그래밍

* 언패킹 관련
  * iterable 시 언패킹을 사용할 것
  * 언패킹 시, 숫자 인덱스 (예: `value[0], value[1] = ...`) 는 사용하지 말 것
* 파일 열기, 닫기 시 컨텍스트 매니저 방식인 `with open(...)` 사용
* `key=lambda x: x['key']` 보다, `key=itemgetter('key')` 사용 권장
* 문자열 단순 연결보다는 f-string 사용
* 포맷팅 방식 등은 1가지로 통일 (예: f-string 으로 통일)
* 라이브러리 사용
  * 빈도수 계산 시 `collections` 사용
  * 반복문 처리 시 `itertools` 사용
    * `itertools.chain`, `itertools.tee` 등 
  * 파일의 경로명 조건을 이용한 리스트 추출 시 `glob` 사용
* 🔮 함수의 인자 간소화가 가능한 경우 간소화해야 함 (예: `value_1, value_2, value_3` → dataclass 기반 `values`)
* attribute 접근 방식으로 `getattr` 사용 권장
* `re.sub` 등 정규표현식 사용 시 표현식 string을 `r'...'` 형식으로 사용하도록 함
* `f = lambda x: ...` 보다는 `def f(x): return ...` 를 사용한다.
* prefix, suffix 검사 시 `startswith()`, `endswith()` 를 사용한다.

### 1-5. 예외 및 오류 처리

* 예외를 삼키는 경우가 없어야 함
* 예외의 종류 (`OOOError` 등) 를 가급적 구체적으로 명시해야 함
* 함수 관련 오류 방지
  * 함수의 인수를 변경 가능한 default value (예: `data: dict = {...}`) 로 하지 않는다.
* assertion을 `try-except` 등을 이용한 제어 메커니즘으로 사용하지 않는다.
* Python 예약어를 변수명으로 사용하지 않는다.

### 1-6. 코드 응집성, 클래스화 등

* 🔮 다음과 같은 경우, 객체 지향 class 로 만드는 것을 적극 고려
  * 🔮 상태 관리용 변수가 많은 경우
  * 🔮 동일한 인수 집합을 갖는 함수가 많은 경우
  * 🔮 상태 값으로 판단되는 값을 조건으로 하여 `if-elif-else` 처리가 있는 경우
* 🔮 하나의 모듈 (Python 코드 파일) 내의 함수들은 응집성이 높아야 함
  * 🔮 비슷한 기능을 하는 함수끼리 모듈로 묶거나... 그 외 응집성 높은 방법으로 해야 함
* 객체의 인터페이스 공개용이 아닌 (= 클래스 내부에서만 쓰이는) 속성, 메서드에 접두사 `_` 권장

## 2. PyTorch 관련

* PyTorch 기본 학습 프로세스 준수
  * `model.optimizer.zero_grad()` 포함 및 순서 준수
  * `loss.backward()` 포함 및 순서 준수
  * `model.optimizer.step()` 포함 및 순서 준수
* `model.train()`, `model.eval()` 포함
* inference 시 `with torch.no_grad()` 등 사용
* loss 로깅 시 `loss` 그래프 자체를 참조하는 대신, `loss.item()` 사용
* scheduler 포함 및 `model.scheduler.step()` 으로 업데이트 되고 있음
* Loss Function 및 Activation Function 의 올바른 사용
  * 다중 분류 문제에서 Categorical Cross Entropy Loss 사용
  * 이진 분류 문제에서 Binary Cross Entropy Loss + Sigmoid 조합 사용 (`nn.BCEWithLogitsLoss` 사용 권장)
* 데이터셋 train/valid/test 분리 시, train loader 에 `shuffle=True` 적용
* 재현성을 위한 `torch.manual_seed` 등 적용
