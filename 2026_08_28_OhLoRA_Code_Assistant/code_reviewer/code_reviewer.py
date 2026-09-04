
import glob
from pathlib import Path

import numpy as np

from code_review_items import default_code_review_func


TEST_CASES_DIR = 'test_cases/tc04.py'


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
        return self.code_review_func(py_codes, self.config, self.current_code_path)

    def review_codes(self, code_path: str, except_path: str | None = None) -> dict[str, str]:
        """Review code in code_path (directory or file)."""

        if code_path.endswith('.py'):
            py_file_paths = [code_path]
        else:
            py_file_paths = glob.glob(f'{code_path}/**/*.py', recursive=True)
            if except_path is not None:
                py_file_paths = [p for p in py_file_paths if not p.startswith(except_path)]

        self.current_code_path = code_path
        code_review_results = self._review_codebase(py_file_paths)
        return code_review_results

    def run_test(self) -> None:
        """Test code reviewer using test cases."""

        total = len(self.test_cases)
        successful = 0

        for test_case in self.test_cases:
            for codebase_to_test, expected_result in test_case.items():
                code_review_result = self.code_review_func(codebase_to_test, self.config)
                if code_review_result == expected_result:
                    successful += 1

        ratio = f'{successful / total * 100:.2f}%'

        print('test result')
        print(f'total         : {total}')
        print(f'successful    : {successful}')
        print(f'success ratio : {ratio}')


# TODO: remove for production
# TODO: result = item.get('info', {}).get('ctx', None) 형태로 수정 및 검증
# TODO: 전체 완료후, 전체 코드리뷰 결과 텍스트파일 저장 -> code_review_items.py 분리 -> 코드리뷰 재실시 -> 결과 비교 -> 차이 수정
"""
for name in checks:
    print(getattr(self, f'_check_{name}')())
"""


class TempTextEmbeddingModel:
    def __init__(self):
        pass

    def get_similarity(self, text1: str, text2: str) -> float:
        return 0.7

    def get_prob(self, text) -> float:
        return 0.7

    def get_embedding(self, text):
        return np.array([[1.0, 0.0, -0.5, 1.4, 0.6, 0.7]])


if __name__ == '__main__':
    text_embeddimg_models = {''}
    code_reviewer = CodeReviewer(code_review_func=default_code_review_func,
                                 text_embedding_models={'default': TempTextEmbeddingModel()})
    code_reviewer.review_codes(TEST_CASES_DIR)
    print(code_reviewer)
