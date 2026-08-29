
import glob
from pathlib import Path


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

    def _review_py_file(self, py_file_path: str) -> dict[str, bool]:
        """Review python code file."""

        py_code = Path(py_file_path).read_text(encoding='utf-8')
        return self.code_review_func(py_code, self.config)

    def review_codes(self, code_path: str) -> dict[dict[str, bool]]:
        """Review code in code_path (directory or file)."""

        if code_path.endswith('.py'):
            py_files = [code_path]
        else:
            py_files = glob.glob('**/*.py', recursive=True)

        code_review_results = {py_file_path: self._review_py_file(py_file_path)
                               for py_file_path in py_files}
        return code_review_results

    def run_test(self) -> None:
        """Test code reviewer using test cases."""

        total = len(self.test_cases)
        successful = 0

        for test_case in self.test_cases:
            for code_to_test, expected_result in test_case.items():
                code_review_result = self.code_review_func(code_to_test, self.config)
                if code_review_result == expected_result:
                    successful += 1

        ratio = f'{successful / total * 100:.2f}%'

        print('test result')
        print(f'total         : {total}')
        print(f'successful    : {successful}')
        print(f'success ratio : {ratio}')


if __name__ == '__main__':
    code_reviewer = CodeReviewer(code_review_func=None, text_embedding_models=None)
    print(code_reviewer)
