
import os
import time
import glob
import gc
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from code_review_items import default_code_review_func


TEST_CASES_DIR = 'test_cases'
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

GTE_MODERNBERT_BASE = 'Alibaba-NLP/gte-modernbert-base'
F2LLM_V2_330M = 'codefuse-ai/F2LLM-v2-330M'
LATEON_CODE_PRETRAIN = 'lightonai/LateOn-Code-pretrain'
EMBEDDING_INFERENCE_LOG_PATH = os.path.join(PROJECT_DIR_PATH, "code_reviewer", "embedding_log.csv")

HIDDEN_SIZE = {GTE_MODERNBERT_BASE: 768,
               F2LLM_V2_330M: 896,
               LATEON_CODE_PRETRAIN: 768}

embedding_log = {
    'task_id': [],
    'func_name': [],
    'text1': [],
    'text2': [],
    'result': [],
    'inference_time': []
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

    def _review_codebase(self, py_file_paths: list[str]) -> dict[str, str]:
        """Review python code file."""

        py_codes = {py_file_path: Path(py_file_path).read_text(encoding='utf-8')
                    for py_file_path in py_file_paths}
        return self.code_review_func(py_codes, self.config, self.current_code_path, self.current_except_path)

    def review_codes(self, code_path: str, except_path: str | None = None) -> dict[str, str]:
        """Review code in code_path (directory or file)."""

        if code_path.endswith('.py'):
            py_file_paths = [code_path]
        else:
            py_file_paths = glob.glob(f'{code_path}/**/*.py', recursive=True)
            if except_path is not None:
                py_file_paths = [p for p in py_file_paths if not p.startswith(except_path)]

        self.current_code_path = code_path
        self.current_except_path = except_path

        code_review_results = self._review_codebase(py_file_paths)
        return code_review_results

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


# TODO: ruff 대체된 부분을 code_reviewer/code_review_standard.md 에 추가
# TODO: line 번호에 따라 최종 결과 출력 정렬
# TODO: remove TempTextEmbeddingModel for production
# TODO: 전체 완료후, 전체 코드리뷰 결과 텍스트파일 저장 -> code_review_items.py 분리 -> 코드리뷰 재실시 -> 결과 비교 -> 차이 수정
# TODO: _get_function_name_by_line, _get_class_name_by_line 통합
# TODO: ast.ClassDef: 'class' -> 'class_def'로 수정
"""
for name in checks:
    print(getattr(self, f'_check_{name}')())
"""


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextEmbeddingModelForInference:
    def __init__(self, model_path: str, hidden_size: int, task_id: str, max_len: int = 256, device: str = 'cuda'):
        self.model_path = model_path
        self.predictor = None
        self.tokenizer = None
        self.device = device
        self.task_id = task_id

        self.max_len = max_len
        self.hidden_size = hidden_size
        self.final_linear = nn.Linear(hidden_size, 1)

    def _append_to_embedding_log(self, func_name: str, text1: str, text2: str, result, inference_time: float):
        embedding_log['task_id'].append(self.task_id)
        embedding_log['func_name'].append(func_name)
        embedding_log['text1'].append(text1)
        embedding_log['text2'].append(text2)
        embedding_log['result'].append(result)
        embedding_log['inference_time'].append(round(inference_time, 3))

        embedding_log_df = pd.DataFrame(embedding_log)
        embedding_log_df.to_csv(EMBEDDING_INFERENCE_LOG_PATH)

    def load_model(self):
        if self.predictor is not None:
            print("model already loaded")
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.predictor = AutoModel.from_pretrained(self.model_path,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float32).to(self.device)
        self.final_linear = nn.Linear(self.hidden_size, 1).to(self.device)

        self.predictor.eval()
        self.final_linear.eval()

    def unload_model(self):
        if self.predictor is None:
            print("model not loaded")
            return

        self.predictor = None
        self.tokenizer = None
        self.final_linear = None

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

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.predictor(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pooling(outputs, attention_mask)
            prob = self.final_linear(emb)

        return prob

    def get_similarity(self, text1: str, text2: str) -> float:
        start_at = time.time()

        with torch.no_grad():
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)

        cos_sim = cosine_similarity(emb1, emb2)
        elapsed_time = time.time() - start_at
        self._append_to_embedding_log('get_similarity', text1, text2, cos_sim, elapsed_time)

        return cos_sim

    def get_prob(self, text) -> float:
        start_at = time.time()
        tokenize_result = self._tokenize_text(text)
        input_ids = tokenize_result['input_ids']
        attention_mask = tokenize_result['attention_mask']

        with torch.no_grad():
            prob = self.forward(input_ids, attention_mask)
            prob = prob.cpu().numpy()
            prob = prob[0][0]

        elapsed_time = time.time() - start_at
        self._append_to_embedding_log('get_prob', text, '', prob, elapsed_time)

        return prob

    def get_embedding(self, text: str):
        start_at = time.time()
        tokenize_result = self._tokenize_text(text)

        input_ids = tokenize_result['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = tokenize_result['attention_mask'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.predictor(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pooling(outputs, attention_mask)
            emb = emb.cpu().numpy()
            emb = emb[0]

        elapsed_time = time.time() - start_at
        self._append_to_embedding_log('get_embedding', text, '', str(emb)[:100], elapsed_time)

        return emb


def get_embedding_model(task_id: str):
    if task_id == "01_unnecessary_prints":
        model_name = GTE_MODERNBERT_BASE
    elif task_id.startswith("01_func_docstring"):
        model_name = F2LLM_V2_330M
    else:
        model_name = LATEON_CODE_PRETRAIN

    return TextEmbeddingModelForInference(
        model_path=os.path.join(PROJECT_DIR_PATH, "embedding_model", "models", task_id),
        hidden_size=HIDDEN_SIZE[model_name],
        task_id=task_id
    )


if __name__ == '__main__':
    task_list = [
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
    text_embedding_models = {task_id: get_embedding_model(task_id) for task_id in task_list}

    code_reviewer = CodeReviewer(code_review_func=default_code_review_func,
                                 text_embedding_models=text_embedding_models)
    code_reviewer.review_codes(code_path=TEST_CASES_DIR)
