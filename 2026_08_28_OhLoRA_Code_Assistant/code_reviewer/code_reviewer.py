
import glob
from pathlib import Path

from code_review_items import default_code_review_func


TEST_CASES_DIR = 'test_cases'


class CodeReviewer:
    def __init__(self,
                 code_review_func,
                 text_embedding_models: dict | None = None,
                 test_cases: list[dict[str, dict[str, bool]]] | None = None,
                 max_line_length: int = 120,
                 max_func_lines: int = 100):

        self.config = {
            'max_line_length': max_line_length,
            'max_func_lines': max_func_lines,
            'text_embedding_model': text_embedding_models
        }

        self.code_review_func = code_review_func
        self.test_cases = test_cases

    def _review_codebase(self, py_file_paths: list[str]) -> dict[str, bool]:
        """Review python code file."""

        py_codes = {py_file_path: Path(py_file_path).read_text(encoding='utf-8')
                    for py_file_path in py_file_paths}
        return self.code_review_func(py_codes, self.config)

    def review_codes(self, code_path: str, except_path: str | None = None) -> dict[str, bool]:
        """Review code in code_path (directory or file)."""

        if code_path.endswith('.py'):
            py_file_paths = [code_path]
        else:
            py_file_paths = glob.glob('**/*.py', recursive=True)
            if except_path is not None:
                py_file_paths = [p for p in py_file_paths if not p.startswith(except_path)]

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


if __name__ == '__main__':
    code_reviewer = CodeReviewer(code_review_func=default_code_review_func, text_embedding_models=None)
    code_reviewer.review_codes(TEST_CASES_DIR)
    print(code_reviewer)
