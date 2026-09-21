
import glob
import gc
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from code_review_items import default_code_review_func


TEST_CASES_DIR = 'test_cases'


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
    def __init__(self, model_path: str, hidden_size: int, max_len: int = 256, device: str = 'cuda'):
        self.model_path = model_path
        self.predictor = None
        self.tokenizer = None
        self.device = device

        self.max_len = max_len
        self.hidden_size = hidden_size
        self.final_linear = nn.Linear(hidden_size, 1)

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
        outputs = self.predictor(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pooling(outputs, attention_mask)
        prob = self.final_linear(emb)
        return prob

    def get_similarity(self, text1: str, text2: str) -> float:
        tokenize_result_text1 = self._tokenize_text(text1)
        tokenize_result_text2 = self._tokenize_text(text2)
        emb1 = self.get_embedding(tokenize_result_text1)
        emb2 = self.get_embedding(tokenize_result_text2)

        return cosine_similarity(emb1, emb2)

    def get_prob(self, text) -> float:
        tokenize_result = self._tokenize_text(text)
        input_ids = tokenize_result['input_ids']
        attention_mask = tokenize_result['attention_mask']
        prob = self.forward(input_ids, attention_mask)

        return prob

    def get_embedding(self, tokenize_result: dict):
        input_ids = tokenize_result['input_ids']
        attention_mask = tokenize_result['attention_mask']
        outputs = self.predictor(input_ids=input_ids, attention_mask=attention_mask)

        emb = mean_pooling(outputs, attention_mask)
        return emb


if __name__ == '__main__':
    text_embeddimg_models = {''}
    code_reviewer = CodeReviewer(code_review_func=default_code_review_func,
                                 text_embedding_models={})
    code_reviewer.review_codes(code_path=TEST_CASES_DIR)
