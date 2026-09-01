import os
import sys
import importlib.metadata

import re
import keyword
import builtins
from difflib import SequenceMatcher
from operator import itemgetter

from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict
from itertools import chain
from ast_utils import parse_py_code

PRESERVED_WORDS = set(keyword.kwlist) | set(dir(builtins))


def simplify_code(original_code: str) -> str:
    result = re.sub(r'\d+(?:\.\d+)?', '0', original_code)
    result = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
                    lambda m: m.group(0) if m.group(0) in PRESERVED_WORDS else "name",
                    result)

    result = re.sub(r'""".*\"""', 'doc', result)
    result = re.sub(r"'''.*\'''", 'doc', result)
    result = re.sub(r'"[^"]*"', 'str', result)
    result = re.sub(r"'[^']*'", 'str', result)

    while '  ' in result:
        result = result.replace('  ', ' ')
    result = result.strip().replace('\n', ';')

    return result


def convert_to_human_friendly_review(final_result_dict: dict[dict[list]]) -> str:
    """Convert json-like format review result into human-friendly review style."""

    final_review = ''

    for py_file_path in final_result_dict:
        human_friendly_review = ''

        for info_key in final_result_dict[py_file_path].keys():
            human_friendly_info_key = info_key or '(최상위 레벨)'

            result = "\n".join(f"   - line {item['line']} 에 있는 {item['name']}"
                               for item in final_result_dict[py_file_path][info_key])
            if result:
                human_friendly_review += f' - 함수: {human_friendly_info_key}\n{result}\n\n'

        if human_friendly_review:
            final_review += f'파일: {py_file_path}\n{human_friendly_review}'

    return final_review


def ellipse_str(original_str: str) -> str:
    if len(original_str) >= 40:
        return f'{original_str[:16]}  ...  {original_str[-16:]}'
    return original_str


class DefaultCodeChecker:
    def __init__(self, py_codes: dict[str, str], config: dict):
        self.py_codes = py_codes
        self.max_line_length = config.get('max_line_length')
        self.max_func_lines = config.get('max_func_lines')
        self.code_indent = config.get('code_indent')
        self.text_embedding_models = config.get('text_embedding_models')

        self.python_libraries = set(list(sys.stdlib_module_names))
        self.third_party_libraries = set(dist.metadata['Name'] for dist in importlib.metadata.distributions())

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

    def _get_definitions_and_usages(self, py_file_path: str, parsed_py_code: list[dict],
                                    imported_dict: dict[list] | None = None) -> tuple[dict[list], dict[list]]:

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

                if imported_dict is not None:
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

        return defined_info, used_info

    def _get_constants(self, py_file_path: str, parsed_py_code: list[dict]) -> dict[list]:
        constant_info = defaultdict(list)

        for item in parsed_py_code:
            line_no = item['line']

            func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
            info_key = func_name or ''

            if item['type_name'] == 'constant':
                constant_with_type = {'name': item['info']['value'], 'type': 'constant', 'line': line_no}
                constant_info[info_key].append(constant_with_type)

        return constant_info

    def _get_function_bodies(self) -> dict[list]:
        function_bodies_info = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            if not parsed_py_code:
                continue

            for item in parsed_py_code:
                if item['type_name'] == 'function_def':
                    function_name = item['info']['name']
                    function_body = item['info']['body']
                    start_line_no = item['info']['start_line']
                    function_bodies_info[py_file_path].append({'name': function_name,
                                                               'body': function_body,
                                                               'start_line': start_line_no})

        return dict(function_bodies_info)

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

    def _is_function_bodies_similar(self, body_1: str, body_2: str) -> bool:
        body_1_lines = [line.strip() for line in body_1.split(';')]
        body_2_lines = [line.strip() for line in body_2.split(';')]

        seq_matcher = SequenceMatcher(None, body_1_lines, body_2_lines)
        lcs = seq_matcher.find_longest_match(0, len(body_1_lines), 0, len(body_2_lines))

        cond_lcs_1 = lcs.size >= 7 and lcs.size >= 0.5 * min(len(body_1_lines), len(body_2_lines))
        cond_lcs_2 = lcs.size >= 3 and lcs.size >= 0.75 * min(len(body_1_lines), len(body_2_lines))

        cond_first_last_1 = (len(body_1_lines) >= 4 and
                             (body_1_lines[:4] == body_2_lines[:4] or body_1_lines[-4:] == body_2_lines[-4:]))
        cond_first_last_2 = (len(body_1_lines) >= 3 and
                             ((body_1_lines[:3] == body_2_lines[:3] and sum(map(len, body_1_lines[:3])) >= 80)
                             or (body_1_lines[-3:] == body_2_lines[-3:] and sum(map(len, body_1_lines[-3:])) >= 80)))

        return cond_lcs_1 or cond_lcs_2 or cond_first_last_1 or cond_first_last_2

    def _check_unused(self) -> str:
        final_result_dict = defaultdict(dict)
        imported_dict = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            defined_info, used_info = self._get_definitions_and_usages(py_file_path, parsed_py_code, imported_dict)
            all_unused_list = defaultdict(list)

            for k in set(defined_info).union(used_info):
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

    def _check_unnecessary_prints(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model = self.text_embedding_models.get('default')

        final_result_dict = defaultdict(dict)
        re_logger = r'^logger\.(debug|info|warning|error|critical)\(.*\)$'
        re_print = r'^print\(.*\)$'

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            py_code_lines = py_code.split('\n')
            logger_loggings = [(line_no + 1, line) for line_no, line
                               in enumerate(py_code_lines) if re.match(re_logger, line)]
            prints = [(line_no + 1, line) for line_no, line
                      in enumerate(py_code_lines) if re.match(re_print, line)]

            print_types = (
                (logger_loggings, 'logging'),
                (prints, 'print')
            )

            for print_collection, print_type in print_types:
                for line_no, line in print_collection:
                    if text_embedding_model.get_prob(line) >= 0.5:
                        func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                        final_result_dict[py_file_path][func_name].append({'name': ellipse_str(line),
                                                                           'type': print_type,
                                                                           'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_duplicates(self) -> str:
        final_result_dict = defaultdict(dict)
        defined_constant_names = set()
        repeated_long_strs = set()
        simplified_function_body_list = []

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            defined_info, used_info = self._get_definitions_and_usages(py_file_path, parsed_py_code)
            constant_value_info = self._get_constants(py_file_path, parsed_py_code)

            defind_constants_info = {func: [info for info in info_list
                                            if info['name'].isupper() and info['type'] != 'import']
                                     for func, info_list in defined_info.items()}
            long_constant_value_info = {func: [info for info in info_list if len(str(info['name'])) >= 8]
                                        for func, info_list in constant_value_info.items()}

            for func_name, info_list in defind_constants_info.items():
                for info in info_list:
                    if info['name'] in defined_constant_names:
                        final_result_dict[py_file_path][func_name].append(info)
                    defined_constant_names.add(info['name'])

            for func_name, info_list in long_constant_value_info.items():
                for info in info_list:
                    if info['name'] in repeated_long_strs:
                        final_result_dict[py_file_path][func_name].append(info)
                    repeated_long_strs.add(info['name'])

        function_bodies_info = self._get_function_bodies()

        for py_file_path, function_body_items in function_bodies_info.items():
            for item in function_body_items:
                function_name = item['name']
                function_body = item['body']
                start_line = item['start_line']

                simplified_body = simplify_code(str(function_body))
                for existing_simplified_body in simplified_function_body_list:
                    if self._is_function_bodies_similar(existing_simplified_body['body'], simplified_body):
                        final_result_dict[py_file_path][function_name].append({'name': function_name,
                                                                               'type': 'func',
                                                                               'line': start_line})
                        break

                simplified_function_body_list.append({'name': function_name,
                                                      'body': simplified_body,
                                                      'start_line': start_line})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _find_all_similar_text_pairs(self, text_embedding_model, value_dict: dict[dict[list]],
                                     only_same: bool = False,
                                     include_same: bool = False) -> list[dict]:
        text_embedding_logs = []
        all_similar_text_pairs = []

        def check_text_similar_with_prevs(info, py_file_path):
            embedding_vector = text_embedding_model.get_embedding(info['name'])
            info['embedding'] = embedding_vector
            info['py_file_path'] = py_file_path

            for log in text_embedding_logs:
                if only_same:
                    if log['name'] == info['name']:
                        all_similar_text_pairs.append(info)
                        break
                else:
                    cos_sim = cosine_similarity(log['embedding'], embedding_vector)
                    if (include_same or log['name'] != info['name']) and cos_sim >= 0.95:
                        all_similar_text_pairs.append(info)
                        break

            text_embedding_logs.append(info)

        for py_file_path in value_dict.keys():
            for func_name, info_list in value_dict[py_file_path].items():
                for info in info_list:
                    check_text_similar_with_prevs(info, py_file_path)

        return all_similar_text_pairs

    def _check_similar_variables(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model = self.text_embedding_models.get('default')

        final_result_dict = defaultdict(dict)
        all_variables_dict = defaultdict(dict)

        for py_file_path in self.parsed_py_codes.keys():
            final_result_dict[py_file_path] = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            defined_info, _ = self._get_definitions_and_usages(py_file_path, parsed_py_code)

            variable_info = {func: [info for info in info_list if info['type'] == 'name']
                             for func, info_list in defined_info.items()}
            all_variables_dict[py_file_path] = variable_info

        all_similar_text_pairs = self._find_all_similar_text_pairs(text_embedding_model, all_variables_dict)

        for info in all_similar_text_pairs:
            line_no = info['line']
            py_file_path = info['py_file_path']

            func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
            final_result_dict[py_file_path][func_name].append({'name': info['name'],
                                                               'type': 'name',
                                                               'line': info['line']})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_same_func_args(self) -> str:
        func_annot_dict = {}

        final_result_dict = defaultdict(dict)
        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            function_defs = [item for item in parsed_py_code if item['type_name'] == 'function_def']

            for item in function_defs:
                func_args = item['info']['args']
                line_no = item['line']

                if func_args:
                    func_arg_names = func_args['name']
                    func_annots = func_args['annot']

                    for arg_name, annot in zip(func_arg_names, func_annots):
                        existing_annot = func_annot_dict.get(arg_name)

                        if existing_annot and existing_annot != annot:
                            func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                            final_result_dict[py_file_path][func_name].append({'name': arg_name,
                                                                               'type': 'arg',
                                                                               'line': line_no})
                        func_annot_dict[arg_name] = annot

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_names(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model = self.text_embedding_models.get('default')

        final_result_dict = defaultdict(dict)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            if not parsed_py_code:
                continue

            final_result_dict[py_file_path] = defaultdict(list)
            var_names = [{'line': item['line'], 'name': item['info']['name']} for item in parsed_py_code
                         if item['type_name'] == 'name' and item['info']['ctx'] == 'Store']
            func_names = [{'line': item['line'], 'name': item['info']['name']} for item in parsed_py_code
                          if item['type_name'] == 'function_def']
            names = chain(var_names, func_names)

            for name_info in names:
                line_no = name_info['line']
                name = name_info['name']

                if text_embedding_model.get_prob(name) >= 0.5:
                    func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                    final_result_dict[py_file_path][func_name].append({'name': name,
                                                                       'type': 'var_or_func',
                                                                       'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_return_matched_with_func_name(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model = self.text_embedding_models.get('default')

        final_result_dict = defaultdict(dict)
        py_file_paths = self.parsed_py_codes.keys()
        py_file_names = [os.path.splitext(os.path.basename(path))[0] for path in py_file_paths]
        py_file_names_set = set(py_file_names)
        return_pattern_dict = {}

        for py_file_path, parsed_py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            if not parsed_py_code:
                continue

            func_names_info = [{'line': item['line'], 'name': item['info']['name']} for item in parsed_py_code
                               if item['type_name'] == 'function_def']
            imported_names_info = [{'line': item['line'], 'name': [info['name'] for info in item['info']['import_names']]}
                                   for item in parsed_py_code
                                   if item['type_name'] == 'import_from' and item['info']['mod'] in py_file_names_set]

            func_names = [info['name'] for info in func_names_info]
            imported_names = [info['name'] for info in imported_names_info]
            func_and_imported_names = list(chain(func_names, *imported_names))

            func_list = '|'.join(map(re.escape, func_and_imported_names))
            pattern = rf'^\s*([a-zA-Z_]\w*)\s*=\s*({func_list})\s*\(.*\)'

            py_file_path_ = re.sub(r'\\+', ' ', py_file_path)
            return_pattern_dict[py_file_path_] = pattern

        for py_file_path, py_code in self.py_codes.items():
            py_file_path_ = re.sub(r'\\+', ' ', py_file_path)
            pattern = return_pattern_dict.get(py_file_path_, None)
            if pattern is None:
                continue

            for line_no, line in enumerate(py_code.split('\n')):
                match = re.match(pattern, line)
                if match:
                    var_name = match.group(1)
                    func_name = match.group(2)

                    if text_embedding_model.get_similarity(var_name, func_name) < 0.5:
                        func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                        final_result_dict[py_file_path][func_name].append({'name': f'{var_name} = {func_name}(...)',
                                                                           'type': 'func_return',
                                                                           'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_library_orders(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            third_party_imported = False
            local_imported = False

            print(py_file_path)
            if not parsed_py_code:
                continue

            parsed_imports = [item for item in parsed_py_code if item['type_name'] in ['import', 'import_from']]
            lib_names_import = [{'line': item['line'], 'name': item['info']['import_names']}
                                for item in parsed_imports
                                if item['type_name'] == 'import']
            lib_names_import = [{'line': item['line'], 'names': [info['name'] for info in item['name']]}
                                for item in lib_names_import]

            lib_names_import_from = [{'line': item['line'], 'names': [item['info']['mod']]}
                                     for item in parsed_imports
                                     if item['type_name'] == 'import_from']

            imported_lib_names = list(chain(lib_names_import, lib_names_import_from))
            imported_lib_names.sort(key=itemgetter('line'))

            for info in imported_lib_names:
                line_no = info['line']
                func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]

                for name in info['names']:
                    if (third_party_imported or local_imported) and name in self.python_libraries:
                        final_result_dict[py_file_path][func_name].append({'name': f'import of builtin "{name}"',
                                                                           'type': 'wrong_builtin_import',
                                                                           'line': line_no})

                    if local_imported and name in self.third_party_libraries:
                        final_result_dict[py_file_path][func_name].append({'name': f'import of 3rd-party "{name}"',
                                                                           'type': 'wrong_third_party_import',
                                                                           'line': line_no})

                    if name in self.third_party_libraries:
                        third_party_imported = True
                    elif name in self.python_libraries:
                        pass
                    else:
                        local_imported = True

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def run_code_review(self) -> dict[str, str]:
        check_unused_review = self._check_unused()
        check_unnecessary_prints_review = self._check_unnecessary_prints()
        check_duplicates_review = self._check_duplicates()
        check_similar_variables_review = self._check_similar_variables()
        check_same_func_args_review = self._check_same_func_args()
        check_names_review = self._check_names()
        check_return_matched_with_func_name_review = self._check_return_matched_with_func_name()
        check_library_orders_review = self._check_library_orders()
        print(check_library_orders_review)


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
