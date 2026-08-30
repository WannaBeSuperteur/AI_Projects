
from ast_utils import parse_py_code


class DefaultCodeChecker:
    def __init__(self, py_codes: dict[str, str], config: dict):
        self.py_codes = py_codes
        self.max_line_length = config.get('max_line_length')
        self.max_func_lines = config.get('max_func_lines')
        self.text_embedding_model = config.get('text_embedding_model')

    def run_code_review(self) -> dict[str, bool]:
        raise NotImplementedError


class PythonBasicsChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)


class PythonBasicConventionChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)


class PythonSimplificationChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)


class PythonOtherPythonicChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)


class PythonExceptionsChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)


class PythonCohesivenessAndClassChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)


class PyTorchChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)


class EntireCodeChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()

        self.python_basics_checker = PythonBasicsChecker(py_codes, config)
        self.python_basic_convention_checker = PythonBasicConventionChecker(py_codes, config)
        self.python_simplification_checker = PythonSimplificationChecker(py_codes, config)
        self.python_other_pythonic_checker = PythonOtherPythonicChecker(py_codes, config)
        self.python_exceptions_checker = PythonExceptionsChecker(py_codes, config)
        self.python_cohesiveness_and_class_checker = PythonCohesivenessAndClassChecker(py_codes, config)
        self.pytorch_checker = PyTorchChecker(py_codes, config)

    def _parse_codes(self):
        self.parsed_py_codes = {py_file_path: parse_py_code(py_code)
                                for py_file_path, py_code in self.py_codes.items()}

    def _check_python_basics(self) -> dict[str, bool]:
        return self.python_basics_checker.run_code_review()

    def _check_basic_convention(self) -> dict[str, bool]:
        return self.python_basic_convention_checker.run_code_review()

    def _check_simplification(self) -> dict[str, bool]:
        return self.python_simplification_checker.run_code_review()

    def _check_other_pythonic(self) -> dict[str, bool]:
        return self.python_other_pythonic_checker.run_code_review()

    def _check_exceptions(self) -> dict[str, bool]:
        return self.python_exceptions_checker.run_code_review()

    def _check_cohesiveness_and_class(self) -> dict[str, bool]:
        return self.python_cohesiveness_and_class_checker.run_code_review()

    def _check_pytorch(self) -> dict[str, bool]:
        return self.pytorch_checker.run_code_review()

    def run_code_review(self) -> dict[str, bool]:
        python_basics_result = self._check_python_basics()
        basic_convention_result = self._check_basic_convention()
        simplification_result = self._check_simplification()
        other_pythonic_result = self._check_other_pythonic()
        exceptions_result = self._check_exceptions()
        cohesiveness_and_class_result = self._check_cohesiveness_and_class()
        pytorch_result = self._check_pytorch()

        final_result = {**python_basics_result,
                        **basic_convention_result,
                        **simplification_result,
                        **other_pythonic_result,
                        **exceptions_result,
                        **cohesiveness_and_class_result,
                        **pytorch_result}
        return final_result


def default_code_review_func(py_codes: dict[str, str], config: dict) -> dict[str, bool]:
    """Default code review function for Oh-LoRA 👱‍♀️ Code Assistant."""

    default_code_checker = EntireCodeChecker(py_codes=py_codes, config=config)
    return default_code_checker.run_code_review()
