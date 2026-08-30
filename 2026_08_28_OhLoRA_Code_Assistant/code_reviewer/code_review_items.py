
from collections import defaultdict
from ast_utils import parse_py_code


def convert_to_human_friendly_review(final_result_dict: dict[dict[list]]) -> str:
    """Convert json-like format review result into human-friendly review style."""

    final_review = ''

    for py_file_path in final_result_dict:
        human_friendly_review = ''

        for info_key in final_result_dict[py_file_path].keys():
            human_friendly_info_key = info_key or '(최상위 레벨)'

            result = "\n".join(f"   - line {item['line']} 에 있는 {item['name']}"
                               for item in final_result_dict[py_file_path][info_key])
            result = result or '   - (발견된 사항 없음)'
            human_friendly_review += f' - 함수: {human_friendly_info_key}\n{result}\n\n'

        human_friendly_review = human_friendly_review or ' - (발견된 사항 없음)'
        final_review += f'파일: {py_file_path}\n{human_friendly_review}'

    return final_review


class DefaultCodeChecker:
    def __init__(self, py_codes: dict[str, str], config: dict):
        self.py_codes = py_codes
        self.max_line_length = config.get('max_line_length')
        self.max_func_lines = config.get('max_func_lines')
        self.text_embedding_model = config.get('text_embedding_model')

    def _parse_codes(self):
        self.parsed_py_codes = {py_file_path: parse_py_code(py_code)
                                for py_file_path, py_code in self.py_codes.items()}

    def _get_function_name_by_line(self):
        self.function_name_by_line_for_codebase = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            if not parsed_py_code:
                continue

            max_line_no = max(item['line'] for item in parsed_py_code)
            function_name_by_line = ['' for _ in range(max_line_no + 1)]

            for item in parsed_py_code:
                if item['type_name'] == 'function_def':
                    start_line_no = item['info']['start_line']
                    end_line_no = item['info']['end_line']

                    for i in range(start_line_no, end_line_no + 1):
                        function_name_by_line[i] = item['info']['name']

            self.function_name_by_line_for_codebase[py_file_path] = function_name_by_line

    def run_code_review(self) -> dict[str, str]:
        raise NotImplementedError


class PythonBasicsChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()
        self._get_function_name_by_line()

    def _delete_imported_review_items(self, file_name_include: str, item_name: str | None):
        for py_file_path in list(self.final_result_dict.keys()):
            file_name_format = f'{file_name_include}.py'
            if not py_file_path.endswith(file_name_format):
                continue

            if item_name is None:  # import ...
                del self.final_result_dict[py_file_path]
            elif item_name == '*':  # from ... import *
                del self.final_result_dict[py_file_path]
            else:  # from ... import ...
                self.final_result_dict[py_file_path][''] = [item for item in self.final_result_dict[py_file_path]['']
                                                            if item['name'] != item_name]

    def _mark_used_as_imported(self):
        for py_file_path, import_info in self.imported_dict.items():
            for info in import_info:
                if info['from'] is not None:  # from ... import ...
                    self._delete_imported_review_items(file_name_include=info['from'], item_name=info['name'])
                else:
                    self._delete_imported_review_items(file_name_include=info['name'], item_name=None)

    def _check_unused(self) -> str:
        final_result_dict = defaultdict(dict)
        imported_dict = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            defined_info = defaultdict(list)
            used_info = defaultdict(list)

            for item in parsed_py_code:
                line_no = item['line']

                func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                func_name_for_func_def = self.function_name_by_line_for_codebase[py_file_path][line_no - 1]

                info_key = func_name or ''
                info_key_for_func_def = func_name_for_func_def or ''

                if item['type_name'] in ['import', 'import_from']:
                    import_infos = [info.get('as_name') or info['name'] for info in item['info']['import_names']
                                    if info['name'] != '*']
                    import_infos_with_type = [{'name': info,
                                               'type': 'import',
                                               'line': line_no} for info in import_infos]
                    defined_info[info_key].extend(import_infos_with_type)

                    imported_dict[py_file_path].extend([{'from': item['info'].get('mod', None), 'name': info['name']}
                                                        for info in import_infos_with_type])

                elif item['type_name'] == 'name':
                    name = item['info']['name']

                    if item['info']['ctx'] == 'Store':
                        defined_info[info_key].append({'name': name,
                                                       'type': 'name',
                                                       'line': line_no})
                    elif item['info']['ctx'] == 'Load':
                        used_info[info_key_for_func_def].append(item['info']['name'])

                elif item['type_name'] == 'function_def':
                    defined_info[info_key_for_func_def].append({'name': item['info']['name'],
                                                                'type': 'func',
                                                                'line': line_no})

                    arg_names = item['info']['args'].get('name', [])
                    arg_names_with_type = [{'name': arg_name,
                                            'type': 'arg',
                                            'line': line_no} for arg_name in arg_names]
                    defined_info[info_key].extend(arg_names_with_type)

            all_unused_list = defaultdict(list)

            for k in list(set(list(defined_info.keys()) + list(used_info.keys()))):
                defined_info_list = dict(defined_info).get(k, [])
                used_names = dict(used_info).get(k, [])
                defined_names = [item['name'] for item in defined_info_list]

                defined_set = set(defined_names)
                used_set = set(used_names)
                unused_set = defined_set - used_set
                unused_list = [item for item in defined_info_list if item['name'] in unused_set]
                all_unused_list[k].extend(unused_list)

            final_result_dict[py_file_path] = dict(all_unused_list)

        self.final_result_dict = final_result_dict
        self.imported_dict = imported_dict
        self._mark_used_as_imported()

        return convert_to_human_friendly_review(final_result_dict)

    def run_code_review(self) -> dict[str, str]:
        check_unused_review = self._check_unused()
        print(check_unused_review)


class PythonBasicConventionChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()


class PythonSimplificationChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()


class PythonOtherPythonicChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()


class PythonExceptionsChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()


class PythonCohesivenessAndClassChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()


class PyTorchChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)
        self._parse_codes()


class EntireCodeChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict):
        super().__init__(py_codes, config)

        self.python_basics_checker = PythonBasicsChecker(py_codes, config)
        self.python_basic_convention_checker = PythonBasicConventionChecker(py_codes, config)
        self.python_simplification_checker = PythonSimplificationChecker(py_codes, config)
        self.python_other_pythonic_checker = PythonOtherPythonicChecker(py_codes, config)
        self.python_exceptions_checker = PythonExceptionsChecker(py_codes, config)
        self.python_cohesiveness_and_class_checker = PythonCohesivenessAndClassChecker(py_codes, config)
        self.pytorch_checker = PyTorchChecker(py_codes, config)

    def _check_python_basics(self) -> dict[str, str]:
        return self.python_basics_checker.run_code_review()

    def _check_basic_convention(self) -> dict[str, str]:
        return self.python_basic_convention_checker.run_code_review()

    def _check_simplification(self) -> dict[str, str]:
        return self.python_simplification_checker.run_code_review()

    def _check_other_pythonic(self) -> dict[str, str]:
        return self.python_other_pythonic_checker.run_code_review()

    def _check_exceptions(self) -> dict[str, str]:
        return self.python_exceptions_checker.run_code_review()

    def _check_cohesiveness_and_class(self) -> dict[str, str]:
        return self.python_cohesiveness_and_class_checker.run_code_review()

    def _check_pytorch(self) -> dict[str, str]:
        return self.pytorch_checker.run_code_review()

    def run_code_review(self) -> dict[str, str]:
        python_basics_result = self._check_python_basics()
        basic_convention_result = self._check_basic_convention()
        simplification_result = self._check_simplification()
        other_pythonic_result = self._check_other_pythonic()
        exceptions_result = self._check_exceptions()
        cohesiveness_and_class_result = self._check_cohesiveness_and_class()
        pytorch_result = self._check_pytorch()

        print('====')
        print(python_basics_result)
        print('====')
        print(basic_convention_result)
        print('====')
        print(simplification_result)
        print('====')
        print(other_pythonic_result)
        print('====')
        print(exceptions_result)
        print('====')
        print(cohesiveness_and_class_result)
        print('====')
        print(pytorch_result)

        final_result = {**python_basics_result,
                        **basic_convention_result,
                        **simplification_result,
                        **other_pythonic_result,
                        **exceptions_result,
                        **cohesiveness_and_class_result,
                        **pytorch_result}
        return final_result


def default_code_review_func(py_codes: dict[str, str], config: dict) -> dict[str, str]:
    """Default code review function for Oh-LoRA 👱‍♀️ Code Assistant."""

    default_code_checker = EntireCodeChecker(py_codes=py_codes, config=config)
    return default_code_checker.run_code_review()
