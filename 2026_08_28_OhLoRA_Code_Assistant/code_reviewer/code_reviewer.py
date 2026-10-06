
import os
import sys
import time
import glob
import gc
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from code_review_items import default_code_review_func

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from embedding_model.train_model import EmbeddingProbPredictor


TEST_CASES_DIR = 'test_cases'

GTE_MODERNBERT_BASE = 'Alibaba-NLP/gte-modernbert-base'
F2LLM_V2_330M = 'codefuse-ai/F2LLM-v2-330M'
LATEON_CODE_PRETRAIN = 'lightonai/LateOn-Code-pretrain'
EMBEDDING_INFERENCE_LOG_PATH = os.path.join(PROJECT_DIR_PATH, "code_reviewer", "embedding_log.csv")

HIDDEN_SIZE = {GTE_MODERNBERT_BASE: 768,
               F2LLM_V2_330M: 896,
               LATEON_CODE_PRETRAIN: 768}

SIMILARITY_TASKS = ['01_similar_variables',
                    '01_return_matched_with_func_name',
                    '01_func_docstring_docstring_and_name',
                    '06_similar_function_names',
                    '02_numeric_values_twice']

embedding_log = {
    'task_id': [],
    'func_name': [],
    'text1': [],
    'text2': [],
    'result': [],
    'inference_time': [],
    'timestamp': []
}

TOP_RULE_COUNT = 5
RULE_NAME_TO_KOREAN = {
    '01_unused': '미 사용 변수/함수 제거',
    '01_unnecessary_prints': '불필요한 print, logging 제거',
    '01_duplicates': '중복되는 사항 common 으로 빼기',
    '01_similar_variables': '유사한 변수명 통일',
    '01_same_func_args': '동일 인자 type 통일',
    '01_names': '의미 있는 함수명, 변수명 사용',
    '01_return_matched_with_func_name': '함수명, 반환값 간 일치',
    '01_library_orders': 'import 순서 준수',
    '01_func_docstring': '함수 docstring, 함수명 의미 일치',
    '01_commented_codes': '주석 처리된 코드 제거',
    '01_empty_file': '빈 파일 TODO 표시 필요',
    '02_const': '고정값 상수화 필요',
    '02_numeric_values': '숫자 값은 위쪽에 상수로',
    '02_line_length': '한 줄의 길이는 일정 글자 이내로',
    '02_files': 'README.md, pyproject.toml 필요',
    '02_functions_length_and_docstring': '100 line 이상 함수 분리, docstring 필요',
    '02_functions_type_hint': '함수 인수 type hint 필요',
    '02_indent': '지나치게 많은 들여쓰기 수정',
    '03_suggest_list_comprehension': '리스트 컴프리헨션 사용이 가능한 경우 사용',
    '03_generator_expression': '제너레이터 표현식 사용이 가능한 경우 사용',
    '03_if_to_dict': 'if-elif-elif-else 구문은 되도록 dict로 수정',
    '03_path_format': '경로를 path/to/file 이 아닌, pathlib 또는 os.path.join 사용',
    '03_defaultdict': '불필요한 변수 생성 대신 defaultdict 권장',
    '03_any_all': '조건문 중첩 대신 any, all 사용',
    '03_zip': 'zip 사용 가능한 경우 사용',
    '03_enumerate': 'enumerate 사용 가능한 경우 사용',
    '03_itertools_product': 'itertools.product 사용 가능한 경우 사용',
    '03_just_read_write_to_read_write_text': '파일 단순 읽기/쓰기는 Path 사용',
    '03_sentence_empty': '비어 있는 문자열 여부 판단 간소화',
    '03_handle_none': 'if a.get("b") ... 형태로 수정 필요',
    '03_extend': '기존 배열의 원소 추가 대신 extend 함수 사용',
    '03_count': '개수 세기에 count 함수 사용',
    '03_index': '인덱스 반환에 index 함수 사용',
    '03_str_join': 'str 단순 += 대신 join 사용',
    '03_use_map': '매우 간결한 변환은 map 사용',
    '04_unpacking_case_1': 'a = my_list[0], b = my_list[1] ... 대신 unpacking 사용 필요',
    '04_unpacking_case_2': '언패킹 시 숫자 인덱스 사용하지 말 것 (변수에 바로 할당)',
    '04_open_file': '파일 열기, 닫기 시 with open(...) 사용',
    '04_key_itemgetter': 'key=itemgetter("key") 사용 권장',
    '04_f_string': '문자열 단순 연결보다는 f-string 사용',
    '04_collections_itertools_glob': '빈도수, 반복문, 경로명 리스트 추출 시 collections, itertools, glob 사용',
    '04_func_args_bindable': '함수의 인자가 하나로 묶을 수 있는 경우 처리 권장',
    '04_attribute_getattr': '함수의 인자가 유동적인 경우 처리 권장',
    '04_regex_r': '정규 표현식 문자열은 r"..." 권장',
    '04_func_lambda': 'f = lambda x: ... 보다는 def f(x): return ... 를 사용',
    '04_prefix_suffix': 'prefix, suffix 검사 시 startswith(), endswith() 사용',
    '05_exception_ignored': '예외를 삼키는 경우가 없어야 함',
    '05_exception_type': '예외의 종류 (OOOError 등) 구체적 명시 권장',
    '05_func_arg_error_prevent': '함수의 인수를 변경 가능한 default value로 하지 않아야 함',
    '05_assertion_try_except': 'assertion을 제어 메커니즘으로 사용하면 안됨',
    '05_python_keywords_args': 'Python 예약어를 변수명으로 사용하지 않아야 함',
    '06_refactor_into_class_case_1_same_args': '동일한 인수 집합을 갖는 함수가 많은 경우 클래스화 고려',
    '06_refactor_into_class_case_2_state_vars_if_else': '상태 값 조건이 있는 if-elif-elif-else 있는 경우 클래스화 고려',
    '06_prefix_for_only_in_class_methods': '클래스 내부에서만 쓰이는 속성, 메서드 (접두사 있음) 호출 비 권장',
    '06_similar_function_names': '유사한 이름의 함수끼리 가까이 위치하도록 수정 권장'
}


class CodeReviewer:
    def __init__(self,
                 code_review_func,
                 text_embedding_models: dict | None = None,
                 test_cases: list[dict[str, dict[str, str]]] | None = None,
                 max_line_length: int = 120,
                 max_func_lines: int = 100,
                 code_indent: int = 4):

        self.config = {
            'max_line_length': max_line_length,
            'max_func_lines': max_func_lines,
            'code_indent': code_indent,
            'text_embedding_models': text_embedding_models
        }

        self.code_review_func = code_review_func
        self.test_cases = test_cases
        self.current_code_path = None

    def _get_files_to_review(self, code_path: str, except_path: str | None = None):
        if code_path.endswith('.py'):
            py_file_paths = [code_path]
        else:
            py_file_paths = glob.glob(f'{code_path}/**/*.py', recursive=True)
            if except_path is not None:
                py_file_paths = [p for p in py_file_paths if not p.startswith(except_path)]

        return py_file_paths

    def _review_codebase(self, py_file_paths: list[str]) -> dict[str, str]:
        """Review python code file."""

        py_codes = {py_file_path: Path(py_file_path).read_text(encoding='utf-8')
                    for py_file_path in py_file_paths}
        return self.code_review_func(py_codes, self.config, self.current_code_path, self.current_except_path)

    def review_codes(self, code_path: str, except_path: str | None = None) -> dict[str, str]:
        """Review code in code_path (directory or file)."""

        py_file_paths = self._get_files_to_review(code_path, except_path)
        self.current_code_path = code_path
        self.current_except_path = except_path

        code_review_results = self._review_codebase(py_file_paths)
        return code_review_results

    def get_file_count(self, code_path: str, except_path: str | None = None) -> int:
        py_file_paths = self._get_files_to_review(code_path, except_path)
        return len(py_file_paths)

    def get_code_lines(self, code_path: str, except_path: str | None = None) -> dict[str, int]:
        py_file_paths = self._get_files_to_review(code_path, except_path)
        code_lines_info = {}

        for file_path in py_file_paths:
            code = Path(file_path).read_text(encoding='utf-8')
            code_lines = len(code.split('\n'))
            code_lines_info[file_path] = code_lines

        return code_lines_info

    def run_test(self) -> None:
        """Test code reviewer using test cases."""

        total = len(self.test_cases)
        successful = 0

        for test_case in self.test_cases:
            for codebase_to_test, expected_result in test_case.items():
                code_review_result = self.code_review_func(codebase_to_test, None, self.config)
                if code_review_result == expected_result:
                    successful += 1

        ratio = f'{successful / total * 100:.2f}%'

        print('test result')
        print(f'total         : {total}')
        print(f'successful    : {successful}')
        print(f'success ratio : {ratio}')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextEmbeddingModelForInference:
    def __init__(self, model_path: str, hidden_size: int, task_id: str, task_type: str,
                 max_len: int = 256, device: str = 'cuda', is_test: bool = False):

        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = device
        self.task_id = task_id

        assert task_type in ['prob', 'cos-sim']
        self.task_type = task_type

        self.max_len = max_len
        self.hidden_size = hidden_size
        self.final_linear = nn.Linear(hidden_size, 1)

        self.is_test = is_test

    def _append_to_embedding_log(self, func_name: str, text1: str, text2: str, result, inference_time: float):
        embedding_log['task_id'].append(self.task_id)
        embedding_log['func_name'].append(func_name)
        embedding_log['text1'].append(text1)
        embedding_log['text2'].append(text2)
        embedding_log['result'].append(result)
        embedding_log['inference_time'].append(round(inference_time, 3))
        embedding_log['timestamp'].append(round(time.time(), 3))

        embedding_log_df = pd.DataFrame(embedding_log)
        embedding_log_df.to_csv(EMBEDDING_INFERENCE_LOG_PATH)

    def load_model(self):
        if self.model is not None:
            print("model already loaded")
            return

        if self.task_type == 'prob':
            base_model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=torch.float32)
            self.model = EmbeddingProbPredictor(base_model=base_model, hidden_size=self.hidden_size)

            all_files = os.listdir(self.model_path)
            model_files = [name for name in all_files if name.endswith('.pth')]
            model_file_path = os.path.join(self.model_path, model_files[0])
            model_state_dict = torch.load(model_file_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(model_state_dict, strict=True)
        else:
            self.model = SentenceTransformer(self.model_path, device=self.device, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        self.model.to(self.device)
        self.model.eval()

    def unload_model(self):
        if self.model is None:
            print("model not loaded")
            return
        self.model = None

        gc.collect()
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

    def _tokenize_text(self, text: str):
        if self.tokenizer is None:
            print("tokenizer not loaded")
            return

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

    def get_similarity(self, text1: str, text2: str) -> float:
        start_at = time.time()

        with torch.no_grad():
            emb1 = self.model.encode(text1)
            emb2 = self.model.encode(text2)
            emb1 = emb1.reshape(1, -1)
            emb2 = emb2.reshape(1, -1)

        cos_sim = cosine_similarity(emb1, emb2)[0][0]

        elapsed_time = time.time() - start_at
        if self.is_test:
            self._append_to_embedding_log('get_similarity', text1, text2, cos_sim, elapsed_time)

        return cos_sim

    def get_prob(self, text) -> float:
        start_at = time.time()
        tokenize_result = self._tokenize_text(text)
        input_ids = tokenize_result['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = tokenize_result['attention_mask'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.model(input_ids, attention_mask)
            prob = torch.sigmoid(prob)
            prob = prob.cpu().numpy()
            prob = prob[0][0]

        elapsed_time = time.time() - start_at
        if self.is_test:
            self._append_to_embedding_log('get_prob', text, '', prob, elapsed_time)

        return prob

    def get_embedding(self, text: str):
        start_at = time.time()

        with torch.no_grad():
            emb = self.model.encode(text)

        elapsed_time = time.time() - start_at
        if self.is_test:
            self._append_to_embedding_log('get_embedding', text, '', str(emb)[:100], elapsed_time)

        return emb


def get_embedding_model(task_id: str):
    if task_id == "01_unnecessary_prints":
        model_name = GTE_MODERNBERT_BASE
    elif task_id.startswith("01_func_docstring"):
        model_name = F2LLM_V2_330M
    else:
        model_name = LATEON_CODE_PRETRAIN

    if task_id in SIMILARITY_TASKS:
        task_type = 'cos-sim'
    else:
        task_type = 'prob'

    return TextEmbeddingModelForInference(
        model_path=os.path.join(PROJECT_DIR_PATH, "embedding_model", "models", task_id),
        hidden_size=HIDDEN_SIZE[model_name],
        task_id=task_id,
        task_type=task_type
    )


def evaluate_code_review_result(code_lines: dict[str, int], item_counts: dict[dict]) -> dict[Any, float]:
    evaluation_result = defaultdict(dict)
    scores = defaultdict(float)
    sum_code_lines = sum(code_lines.values())

    for rule_id, rule_review_result in item_counts.items():
        if '(코드 전체 경로)' in rule_review_result.keys():
            evaluation_result[rule_id] = {'entire_code': max(0, 100 - 20 * rule_review_result['(코드 전체 경로)'])}
            scores[rule_id] = evaluation_result[rule_id]['entire_code'] / 100
        else:
            evaluation_result[rule_id] = {file_path: max(0, code_lines[file_path] - 100 * issue_cnt)
                                          for file_path, issue_cnt in rule_review_result.items()}

            for file_path in code_lines.keys():
                if file_path not in evaluation_result[rule_id]:
                    evaluation_result[rule_id][file_path] = code_lines[file_path]

            scores[rule_id] = sum(evaluation_result[rule_id].values()) / sum_code_lines

    return dict(scores)


def mark_score(score: float) -> str:
    score_mark = f'{round(100 * score, 1)} 점'

    if score >= 0.9:
        return f'{score_mark} 👍'
    elif score >= 0.6:
        return f'{score_mark}'
    else:
        return f'{score_mark} 🚨'


def run_entire_code_review(code_path: str) -> dict:
    task_list_with_embedding = [
        "01_unnecessary_prints",
        "01_similar_variables",
        "01_names",
        "01_return_matched_with_func_name",
        "01_func_docstring_single_responsibility",
        "01_func_docstring_docstring_and_name",
        "04_func_args_bindable",
        "04_func_args_dynamic",
        "06_refactor_into_class_case_2_state_vars_if_else",
        "06_similar_function_names",
        "02_numeric_values_maybe_const",
        "02_numeric_values_twice"
    ]
    text_embedding_models = {task_id: get_embedding_model(task_id) for task_id in task_list_with_embedding}

    code_reviewer = CodeReviewer(code_review_func=default_code_review_func,
                                 text_embedding_models=text_embedding_models)

    code_lines = code_reviewer.get_code_lines(code_path=code_path)
    item_counts, code_review_result = code_reviewer.review_codes(code_path=code_path)

    eval_result = evaluate_code_review_result(code_lines, item_counts)
    eval_result_sorted = list(sorted(eval_result.items(), key=lambda x: x[1]))
    mean_score = np.mean(list(eval_result.values()))

    eval_result_kor = [f'[{RULE_NAME_TO_KOREAN[rule_id]}] : {mark_score(score)}'
                       for rule_id, score in eval_result.items()]
    top_eval_result_kor = {f'[{RULE_NAME_TO_KOREAN[rule_id]}] : {mark_score(score)}'
                           for rule_id, score in eval_result_sorted[:TOP_RULE_COUNT]}

    eval_result_kor_summary = ', \n'.join(eval_result_kor)
    top_items_eval_summary = ', \n'.join(top_eval_result_kor)
    code_review_result_str = '\n'.join([f'{RULE_NAME_TO_KOREAN[rule_id]}:\n{rule_review_result}'
                                        for rule_id, rule_review_result in code_review_result.items()])

    return {'eval_result_kor_summary': eval_result_kor_summary,
            'top_items_eval_summary': top_items_eval_summary,
            'code_review_result_str': code_review_result_str,
            'mean_score': round(100 * mean_score, 1)}


if __name__ == '__main__':
    code_review_result = run_entire_code_review(code_path=TEST_CASES_DIR)

    for k, v in code_review_result.items():
        print(k)
        print(v)
