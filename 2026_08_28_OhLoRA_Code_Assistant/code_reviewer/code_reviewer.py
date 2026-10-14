
import os
import sys
import time
import glob
import gc
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ast_utils import remove_after_py
from code_review_items import default_code_review_func

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from rule_and_model_info import *
from embedding_model.train_model import EmbeddingProbPredictor
from project_common import ENTIRE_CODE_PATH


TEST_CASES_DIR = 'test_cases'
EMBEDDING_INFERENCE_LOG_PATH = os.path.join(PROJECT_DIR_PATH, "code_reviewer", "embedding_log.csv")

SIMILARITY_TASKS = [RULE_SIMILAR_VARIABLES,
                    RULE_RETURN_MATCHED_NAME,
                    RULE_FUNC_DOCSTRING_NAME,
                    RULE_SIMILAR_FUNCTION_NAMES,
                    RULE_NUMERIC_TWICE]

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


class CodeReviewer:
    def __init__(self,
                 code_review_func: Callable,
                 base_config: dict[str, int],
                 text_embedding_models: dict | None = None,
                 test_cases: list[dict[str, dict[str, str]]] | None = None):

        self.config = {
            'max_line_length': base_config['max_line_length'],
            'max_func_lines': base_config['max_func_lines'],
            'code_indent': base_config['code_indent'],
            'text_embedding_models': text_embedding_models
        }

        self.code_review_func = code_review_func
        self.test_cases = test_cases
        self.current_code_path = None

    def _get_files_to_review(self, code_path: str, except_path: str | None = None) -> list[str]:
        """Find all the Python code files to review."""

        if ' ||| ' in code_path:
            py_file_paths = [p for p in code_path.split(' ||| ') if p.endswith('.py')]
        elif code_path.endswith('.py'):
            py_file_paths = [code_path]
        else:
            py_file_paths = glob.glob(f'{code_path}/**/*.py', recursive=True)
            if except_path is not None:
                py_file_paths = [p for p in py_file_paths if not p.startswith(except_path)]

        return py_file_paths

    def _review_codebase(self, py_file_paths: list[str]) -> tuple:
        """Review python code file."""

        py_codes = {py_file_path: Path(py_file_path).read_text(encoding='utf-8')
                    for py_file_path in py_file_paths}
        return self.code_review_func(py_codes, self.config, self.current_code_path, self.current_except_path)

    def review_codes(self, code_path: str, except_path: str | None = None) -> tuple:
        """Review code in code_path (directory or file)."""

        py_file_paths = self._get_files_to_review(code_path, except_path)
        self.current_code_path = code_path
        self.current_except_path = except_path

        item_counts, code_review_result = self._review_codebase(py_file_paths)
        return item_counts, code_review_result

    def get_file_count(self, code_path: str, except_path: str | None = None) -> int:
        py_file_paths = self._get_files_to_review(code_path, except_path)
        return len(py_file_paths)

    def get_code_lines(self, code_path: str, except_path: str | None = None) -> dict[str, int]:
        """Find the number of code lines for each Python code file."""

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


class TextEmbeddingModelForInference:
    def __init__(self, model_config: dict[str, Any], task_config: dict[str, str],
                 max_len: int = 256, device: str = 'cuda', is_test: bool = False):

        self.model_path = model_config['model_path']
        self.model = None
        self.tokenizer = None
        self.device = device
        self.task_id = task_config['task_id']

        self.task_type = task_config['task_type']
        assert self.task_type in ['prob', 'cos-sim']

        self.max_len = max_len
        self.hidden_size = model_config['hidden_size']
        self.final_linear = nn.Linear(self.hidden_size, 1)

        self.is_test = is_test

    def _append_to_embedding_log(self, func_name: str, text1: str, text2: str, result: Any,
                                 inference_time: float) -> None:

        """Add embedding log for test."""

        embedding_log['task_id'].append(self.task_id)
        embedding_log['func_name'].append(func_name)
        embedding_log['text1'].append(text1)
        embedding_log['text2'].append(text2)
        embedding_log['result'].append(result)
        embedding_log['inference_time'].append(round(inference_time, 3))
        embedding_log['timestamp'].append(round(time.time(), 3))

        embedding_log_df = pd.DataFrame(embedding_log)
        embedding_log_df.to_csv(EMBEDDING_INFERENCE_LOG_PATH)

    def load_model(self) -> None:
        """Load embedding models."""

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

    def unload_model(self) -> None:
        """Unload embedding models."""

        if self.model is None:
            print("model not loaded")
            return
        self.model = None

        gc.collect()
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

    def _tokenize_text(self, text: str) -> dict | None:
        """Tokenize the text for embedding models."""

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
        """Find the cosine similarity of two texts."""

        start_at = time.time()

        with torch.no_grad():
            embedding1 = self.model.encode(text1)
            embedding2 = self.model.encode(text2)
            embedding1 = embedding1.reshape(1, -1)
            embedding2 = embedding2.reshape(1, -1)

        cos_sim = cosine_similarity(embedding1, embedding2)[0][0]

        elapsed_time = time.time() - start_at
        if self.is_test:
            self._append_to_embedding_log('get_similarity', text1, text2, cos_sim, elapsed_time)

        return cos_sim

    def get_prob(self, text: str) -> float:
        """Find the probability (of having a specific condition) of the text, using embedding model."""

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
        """Get embedding vector of the text."""

        start_at = time.time()

        with torch.no_grad():
            embedding = self.model.encode(text)

        elapsed_time = time.time() - start_at
        if self.is_test:
            self._append_to_embedding_log('get_embedding', text, '', str(embedding)[:100], elapsed_time)

        return embedding


def get_embedding_model(task_id: str) -> TextEmbeddingModelForInference:
    """Get appropriate embedding model."""

    if task_id == RULE_UNNECESSARY_PRINTS:
        model_name = GTE_MODERNBERT_BASE
    elif task_id.startswith(RULE_FUNC_DOCSTRING):
        model_name = F2LLM_V2_330M
    else:
        model_name = LATEON_CODE_PRETRAIN

    if task_id in SIMILARITY_TASKS:
        task_type = 'cos-sim'
    else:
        task_type = 'prob'

    model_config = {
        'model_path': os.path.join(PROJECT_DIR_PATH, "embedding_model", "models", task_id),
        'hidden_size': HIDDEN_SIZE[model_name]
    }
    task_config = {
        'task_id': task_id,
        'task_type': task_type
    }

    return TextEmbeddingModelForInference(model_config=model_config, task_config=task_config)


def evaluate_code_review_result(code_lines: dict[str, int], item_counts: dict[dict]) -> dict[Any, float]:
    """Evaluate score, using the code review result."""

    evaluation_result = defaultdict(dict)
    scores = defaultdict(float)
    sum_code_lines = sum(code_lines.values())

    for rule_id, rule_review_result in item_counts.items():
        if ENTIRE_CODE_PATH in rule_review_result.keys():
            evaluation_result[rule_id] = {'entire_code': max(0, 100 - 50 * rule_review_result[ENTIRE_CODE_PATH])}
            scores[rule_id] = evaluation_result[rule_id]['entire_code'] / 100
        else:
            weight = 20 if rule_id == '02_indent' else 100
            evaluation_result[rule_id] = {file_path: max(0, code_lines[remove_after_py(file_path)] - weight * issue_cnt)
                                          for file_path, issue_cnt in rule_review_result.items()}

            not_checked_file_paths = [path for path in code_lines.keys() if path not in evaluation_result[rule_id]]
            for file_path in not_checked_file_paths:
                evaluation_result[rule_id][file_path] = code_lines[file_path]

            scores[rule_id] = sum(evaluation_result[rule_id].values()) / sum_code_lines

    rule_id_not_checked = [rule_id for rule_id in RULE_NAME_TO_KOREAN.keys() if rule_id not in item_counts.keys()]
    for rule_id in rule_id_not_checked:
        scores[rule_id] = 1.0

    return dict(scores)


def mark_score(score: float) -> str:
    score_mark = f'{round(100 * score, 1)} 점'

    if score >= 0.9:
        return f'{score_mark} 👍'
    elif score >= 0.6:
        return f'{score_mark}'
    else:
        return f'{score_mark} 🚨'


def run_entire_code_review(code_path: str, except_path: str | None = None) -> dict:
    """Run entire code review of all the Python code files from the given path."""

    text_embedding_models = {task_id: get_embedding_model(task_id) for task_id in RULES_EMBEDDING}

    code_reviewer = CodeReviewer(code_review_func=default_code_review_func,
                                 text_embedding_models=text_embedding_models,
                                 base_config={'max_line_length': 120, 'max_func_lines': 100, 'code_indent': 4})

    code_lines = code_reviewer.get_code_lines(code_path=code_path)
    item_counts, code_review_result = code_reviewer.review_codes(code_path=code_path, except_path=except_path)

    eval_result = evaluate_code_review_result(code_lines, item_counts)
    eval_result_sorted = list(sorted(eval_result.items(), key=lambda x: x[1]))
    mean_score = np.mean(list(eval_result.values()))

    eval_result_kor = [f'[{RULE_NAME_TO_KOREAN[rule_id]}] : {mark_score(score)}'
                       for rule_id, score in eval_result.items()]
    top_eval_result_kor = [f'[{RULE_NAME_TO_KOREAN[rule_id]}] : {mark_score(score)}'
                           for rule_id, score
                           in sorted(eval_result_sorted[:TOP_RULE_COUNT], key=lambda x: x[1], reverse=True)]

    eval_result_kor_summary = ', \n'.join(eval_result_kor)
    top_items_eval_summary = ', \n'.join(top_eval_result_kor)
    code_review_result_str = '\n'.join([f'## 규칙: {RULE_NAME_TO_KOREAN[rule_id]}\n{rule_review_result}'
                                        for rule_id, rule_review_result in code_review_result.items()
                                        if str(rule_review_result).strip()])

    return {'item_counts': item_counts,
            'eval_result_kor_summary': eval_result_kor_summary,
            'top_items_eval_summary': top_items_eval_summary,
            'code_review_result_str': code_review_result_str,
            'mean_score': round(100 * mean_score, 1)}


if __name__ == '__main__':
    code_review_result = run_entire_code_review(code_path=TEST_CASES_DIR)

    for k, v in code_review_result.items():
        if k != 'item_counts':
            print(f'\n\n[ {k} ]')
            print(v)

    print('\n\n[ item_counts ]')
    for rule_id, rule_review_result in code_review_result['item_counts'].items():
        print(rule_id, rule_review_result)
