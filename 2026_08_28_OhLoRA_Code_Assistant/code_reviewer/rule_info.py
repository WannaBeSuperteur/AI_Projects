
# 01. 기본 사항
RULE_UNUSED = '01_unused'
RULE_UNNECESSARY_PRINTS = '01_unnecessary_prints'
RULE_DUPLICATES = '01_duplicates'
RULE_SIMILAR_VARIABLES = '01_similar_variables'
RULE_SAME_FUNC_ARGS = '01_same_func_args'
RULE_NAMES = '01_names'
RULE_RETURN_MATCHED_NAME = '01_return_matched_with_func_name'
RULE_LIB_ORDERS = '01_library_orders'
RULE_FUNC_DOCSTRING = '01_func_docstring'
RULE_FUNC_DOCSTRING_RESPONSIBILITY = '01_func_docstring_single_responsibility'
RULE_FUNC_DOCSTRING_NAME = '01_func_docstring_docstring_and_name'
RULE_COMMENTED_CODES = '01_commented_codes'
RULE_EMPTY_FILE = '01_empty_file'

# 02. 기본 코딩 컨벤션
RULE_CONST = '02_const'
RULE_NUMERIC = '02_numeric_values'
RULE_NUMERIC_CONST = '02_numeric_values_maybe_const'
RULE_NUMERIC_TWICE = '02_numeric_values_twice'
RULE_LINE_LENGTH = '02_line_length'
RULE_FILES = '02_files'
RULE_FUNCTION_LENGTH_AND_DOCSTRING = '02_functions_length_and_docstring'
RULE_FUNCTION_TYPE_HINT = '02_functions_type_hint'
RULE_INDENT = '02_indent'

# 03. Python 문법 간소화
RULE_LIST_COMPREHENSION = '03_suggest_list_comprehension'
RULE_GENERATOR_EXPRESSION = '03_generator_expression'
RULE_IF_TO_DICT = '03_if_to_dict'
RULE_PATH_FORMAT = '03_path_format'
RULE_DEFAULTDICT = '03_defaultdict'
RULE_ANY_ALL = '03_any_all'
RULE_ZIP = '03_zip'
RULE_ENUMERATE = '03_enumerate'
RULE_ITERTOOLS_PRODUCT = '03_itertools_product'
RULE_JUST_READ_WRITE = '03_just_read_write_to_read_write_text'
RULE_SENTENCE_EMPTY_CHECK = '03_sentence_empty'
RULE_HANDLE_NONE = '03_handle_none'
RULE_EXTEND = '03_extend'
RULE_COUNT = '03_count'
RULE_INDEX = '03_index'
RULE_STR_JOIN = '03_str_join'
RULE_USE_MAP = '03_use_map'

# 04. 그 외 Pythonic 한 프로그래밍
RULE_UNPACKING_1 = '04_unpacking_case_1'
RULE_UNPACKING_2 = '04_unpacking_case_2'
RULE_OPEN_FILE = '04_open_file'
RULE_KEY_ITEMGETTER = '04_key_itemgetter'
RULE_F_STRING = '04_f_string'
RULE_COLLECTIONS_ITERTOOLS_GLOB = '04_collections_itertools_glob'
RULE_GETATTR = '04_attribute_getattr'
RULE_REGEX_R = '04_regex_r'
RULE_FUNCTION_LAMBDA = '04_func_lambda'
RULE_PREFIX_SUFFIX = '04_prefix_suffix'
RULE_ARGS_BINDABLE = '04_func_args_bindable'
RULE_ARGS_DYNAMIC = '04_func_args_dynamic'

# 05. 예외 및 오류 처리
RULE_EXCEPTION_IGNORED = '05_exception_ignored'
RULE_EXCEPTION_TYPE = '05_exception_type'
RULE_FUNCTION_ARG_ERROR = '05_func_arg_error_prevent'
RULE_ASSERTION_TRY_EXCEPT = '05_assertion_try_except'
RULE_PYTHON_KEYWORD_ARGS = '05_python_keywords_args'

# 06. 코드 응집성, 클래스화 등
RULE_REFACTOR_SAME_ARGS = '06_refactor_into_class_case_1_same_args'
RULE_REFACTOR_STATE_VARS = '06_refactor_into_class_case_2_state_vars_if_else'
RULE_PREFIX_CLASS_METHODS = '06_prefix_for_only_in_class_methods'
RULE_SIMILAR_FUNCTION_NAMES = '06_similar_function_names'

PROBS = {
    RULE_UNNECESSARY_PRINTS: 0.8,
    RULE_NAMES: 0.1,
    RULE_FUNC_DOCSTRING_RESPONSIBILITY: 0.3,
    RULE_FUNC_DOCSTRING_NAME: 0.3,
    RULE_ARGS_BINDABLE: 0.9,
    RULE_ARGS_DYNAMIC: 0.9,
    RULE_REFACTOR_STATE_VARS: 0.98,
    RULE_NUMERIC_CONST: 0.85
}

SIMILARITIES = {
    RULE_SIMILAR_FUNCTION_NAMES: {'high': 0.95, 'mid': 0.75},
    RULE_NUMERIC_TWICE: 0.8,
    RULE_RETURN_MATCHED_NAME: 0.35,
    RULE_SIMILAR_VARIABLES: 0.93
}

RULE_NAME_TO_KOREAN = {
    RULE_UNUSED: '미 사용 변수/함수 제거',
    RULE_UNNECESSARY_PRINTS: '불필요한 print, logging 제거',
    RULE_DUPLICATES: '중복되는 사항 common 으로 빼기',
    RULE_SIMILAR_VARIABLES: '유사한 변수명 통일',
    RULE_SAME_FUNC_ARGS: '동일 인자 type 통일',
    RULE_NAMES: '의미 있는 함수명, 변수명 사용',
    RULE_RETURN_MATCHED_NAME: '함수명, 반환값 간 일치',
    RULE_LIB_ORDERS: 'import 순서 준수',
    RULE_FUNC_DOCSTRING: '함수 docstring, 함수명 의미 일치',
    RULE_COMMENTED_CODES: '주석 처리된 코드 제거',
    RULE_EMPTY_FILE: '빈 파일 TODO 표시 필요',
    RULE_CONST: '고정값 상수화 필요',
    RULE_NUMERIC: '숫자 값은 위쪽에 상수로',
    RULE_LINE_LENGTH: '한 줄의 길이는 일정 글자 이내로',
    RULE_FILES: 'README.md, pyproject.toml 필요',
    RULE_FUNCTION_LENGTH_AND_DOCSTRING: '100 line 이상 함수 분리, docstring 필요',
    RULE_FUNCTION_TYPE_HINT: '함수 인수 type hint 필요',
    RULE_INDENT: '지나치게 많은 들여쓰기 수정',
    RULE_LIST_COMPREHENSION: '리스트 컴프리헨션 사용이 가능한 경우 사용',
    RULE_GENERATOR_EXPRESSION: '제너레이터 표현식 사용이 가능한 경우 사용',
    RULE_IF_TO_DICT: 'if-elif-elif-else 구문은 되도록 dict로 수정',
    RULE_PATH_FORMAT: '경로를 path/to/file 이 아닌, pathlib 또는 os.path.join 사용',
    RULE_DEFAULTDICT: '불필요한 변수 생성 대신 defaultdict 권장',
    RULE_ANY_ALL: '조건문 중첩 대신 any, all 사용',
    RULE_ZIP: 'zip 사용 가능한 경우 사용',
    RULE_ENUMERATE: 'enumerate 사용 가능한 경우 사용',
    RULE_ITERTOOLS_PRODUCT: 'itertools.product 사용 가능한 경우 사용',
    RULE_JUST_READ_WRITE: '파일 단순 읽기/쓰기는 Path 사용',
    RULE_SENTENCE_EMPTY_CHECK: '비어 있는 문자열 여부 판단 간소화',
    RULE_HANDLE_NONE: 'if a.get("b") ... 형태로 수정 필요',
    RULE_EXTEND: '기존 배열의 원소 추가 대신 extend 함수 사용',
    RULE_COUNT: '개수 세기에 count 함수 사용',
    RULE_INDEX: '인덱스 반환에 index 함수 사용',
    RULE_STR_JOIN: 'str 단순 += 대신 join 사용',
    RULE_USE_MAP: '매우 간결한 변환은 map 사용',
    RULE_UNPACKING_1: 'a = my_list[0], b = my_list[1] ... 대신 unpacking 사용 필요',
    RULE_UNPACKING_2: '언패킹 시 숫자 인덱스 사용하지 말 것 (변수에 바로 할당)',
    RULE_OPEN_FILE: '파일 열기, 닫기 시 with open(...) 사용',
    RULE_KEY_ITEMGETTER: 'key=itemgetter("key") 사용 권장',
    RULE_F_STRING: '문자열 단순 연결보다는 f-string 사용',
    RULE_COLLECTIONS_ITERTOOLS_GLOB: '빈도수, 반복문, 경로명 리스트 추출 시 collections, itertools, glob 사용',
    RULE_ARGS_BINDABLE: '함수의 인자가 하나로 묶을 수 있는 경우 처리 권장',
    RULE_GETATTR: '함수의 인자가 유동적인 경우 처리 권장',
    RULE_REGEX_R: '정규 표현식 문자열은 r"..." 권장',
    RULE_FUNCTION_LAMBDA: 'f = lambda x: ... 보다는 def f(x): return ... 를 사용',
    RULE_PREFIX_SUFFIX: 'prefix, suffix 검사 시 startswith(), endswith() 사용',
    RULE_EXCEPTION_IGNORED: '예외를 삼키는 경우가 없어야 함',
    RULE_EXCEPTION_TYPE: '예외의 종류 (OOOError 등) 구체적 명시 권장',
    RULE_FUNCTION_ARG_ERROR: '함수의 인수를 변경 가능한 default value로 하지 않아야 함',
    RULE_ASSERTION_TRY_EXCEPT: 'assertion을 제어 메커니즘으로 사용하면 안됨',
    RULE_PYTHON_KEYWORD_ARGS: 'Python 예약어를 변수명으로 사용하지 않아야 함',
    RULE_REFACTOR_SAME_ARGS: '동일한 인수 집합을 갖는 함수가 많은 경우 클래스화 고려',
    RULE_REFACTOR_STATE_VARS: '상태 값 조건이 있는 if-elif-elif-else 있는 경우 클래스화 고려',
    RULE_PREFIX_CLASS_METHODS: '클래스 내부에서만 쓰이는 속성, 메서드 (접두사 있음) 호출 비 권장',
    RULE_SIMILAR_FUNCTION_NAMES: '유사한 이름의 함수끼리 가까이 위치하도록 수정 권장'
}
