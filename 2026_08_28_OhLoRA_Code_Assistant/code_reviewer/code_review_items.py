import os
import sys
import io
import importlib.metadata

import re
import tokenize
import keyword
import builtins

from typing import Callable, Optional
from difflib import SequenceMatcher
from operator import itemgetter

from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict
from itertools import chain, product
from ast_utils import parse_py_code

PRESERVED_WORDS = set(keyword.kwlist) | set(dir(builtins))
QUOTES = "'" + '"'
TWO_DOUBLE_QUOTES = '""'


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
    replaced_newline_str = original_str.replace('\n', ' ')
    if len(replaced_newline_str) >= 40:
        return f'{replaced_newline_str[:16]}  ...  {replaced_newline_str[-16:]}'
    return replaced_newline_str


def extract_comment(line):
    try:
        readline = io.StringIO(line).readline
        for token in tokenize.generate_tokens(readline):
            if token.type == tokenize.COMMENT:
                return token.string
    except tokenize.TokenError:
        pass
    return ""


def check_regex_matched_lines(py_code: str, regex: str, except_comment: bool = True,
                              except_docstring: bool = True) -> list[dict[str]]:

    lines = py_code.split('\n')
    lines = [{'line_no': i + 1, 'line': line} for i, line in enumerate(lines)]

    pattern = re.compile(regex)
    matched_lines = [line for line in lines if pattern.search(line['line'])]

    if except_comment:
        matched_lines = [line for line in matched_lines if not line['line'].strip().startswith("# ")]
    if except_docstring:
        matched_lines = [line for line in matched_lines
                         if not line['line'].strip().startswith('"""') and not line['line'].strip().endswith('"""')]

    return matched_lines


def is_valid_code(line: str) -> bool:
    if not line:
        return False
    if line.strip().startswith('#'):  # comment
        return False
    if line.strip().startswith('"""'):  # docstring
        return False
    return True


def check_a_in_b(a: str, b: str) -> bool:
    tokens = re.sub(r'[^a-zA-Z0-9_]', ' ', b).split()
    return a in tokens


class DefaultCodeChecker:
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        self.py_codes = py_codes
        self.max_line_length = config.get('max_line_length')
        self.max_func_lines = config.get('max_func_lines')
        self.code_indent = config.get('code_indent')
        self.text_embedding_models = config.get('text_embedding_models')

        self.code_path = code_path

        self.python_libraries = set(list(sys.stdlib_module_names))
        self.third_party_libraries = set(dist.metadata['Name'] for dist in importlib.metadata.distributions())
        self.final_result_dict = defaultdict(dict)

    def _parse_codes(self):
        self.parsed_py_codes = {py_file_path: parse_py_code(py_code)
                                for py_file_path, py_code in self.py_codes.items()}

    def _get_function_name_by_line(self):
        self.function_name_by_line_for_codebase = defaultdict(list)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            if not parsed_py_code:
                continue

            max_line_no = len(self.py_codes[py_file_path].split('\n')) + 1
            function_name_by_line = ['' for _ in range(max_line_no + 1)]

            for item in parsed_py_code:
                if item['type_name'] == 'function_def':
                    start_line_no = item['info']['start_line']
                    end_line_no = item['info']['end_line']

                    for i in range(start_line_no, end_line_no + 1):
                        function_name_by_line[i] = item['info']['name']

            self.function_name_by_line_for_codebase[py_file_path] = function_name_by_line

        for py_file_path, parsed_py_code in self.py_codes.items():
            py_file_path_ = py_file_path.replace(r"\\", r"\"")

            max_line_no = len(self.py_codes[py_file_path_].split('\n')) + 1
            function_name_by_line = ['' for _ in range(max_line_no + 1)]

            if py_file_path_ not in self.function_name_by_line_for_codebase:
                self.function_name_by_line_for_codebase[py_file_path_] = function_name_by_line

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

    def _add_regex_matched_lines(self, regex: str, match_func: Optional[Callable] = None, type_name: str = 'regex',
                                 forward_window_size: int = 5) -> None:

        for py_file_path, py_code in self.py_codes.items():
            lines = [line.strip() for line in py_code.split('\n')]
            lines = [line[:len(line)-len(extract_comment(line))] for line in lines]

            line_idx = 0
            while line_idx < len(lines):
                line_no = line_idx + 1
                lines_to_search = '\n'.join(lines[line_idx:line_idx+forward_window_size])
                matched = re.match(regex, lines_to_search)

                if matched:
                    matched_text = matched.group(0)
                    matched_groups = list(matched.groups())

                    func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                    lines_to_show = lines_to_search.replace('\n', ' ')
                    if len(lines_to_show) > 40:
                        lines_to_show = lines_to_show[:36] + ' ...'

                    if match_func is None or match_func(matched_groups):
                        self.final_result_dict[py_file_path][func_name].append({'name': lines_to_show,
                                                                                'type': type_name,
                                                                                'line': line_no})
                        line_idx += matched_text.count('\n') + 1
                    else:
                        line_idx += 1
                else:
                    line_idx += 1

    def run_code_review(self) -> dict[str, str]:
        raise NotImplementedError


class PythonBasicsChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
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

        cond_lcs_1 = lcs.size >= 7
        cond_lcs_2 = lcs.size >= 4 and lcs.size >= 0.5 * min(len(body_1_lines), len(body_2_lines))
        cond_lcs_3 = lcs.size >= 3 and lcs.size >= 0.75 * min(len(body_1_lines), len(body_2_lines))

        return cond_lcs_1 or cond_lcs_2 or cond_lcs_3

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
                unused_list = [item for item in defined_info_list if item['name'] in unused_set and item['name'] != '_']
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

        for py_file_path, _ in self.py_codes.items():
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

    def _check_func_docstring(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model_single_responsibility = self.text_embedding_models.get('default')
        text_embedding_model_docstring_and_name = self.text_embedding_models.get('default')

        final_result_dict = defaultdict(dict)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            func_defs = [item for item in parsed_py_code if item['type_name'] == 'function_def']
            func_docstrings_info = [{'line': item['line'],
                                     'name': item['info']['name'],
                                     'docstring': item.get('docstring', None)} for item in func_defs]
            func_docstrings_info = [{'line': item['line'],
                                     'name': item['name'],
                                     'docstring': item['docstring'].replace('\n', ' ')}
                                    for item in func_docstrings_info if item['docstring'] is not None]

            for item in func_docstrings_info:
                line_no = item['line']
                func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]

                if text_embedding_model_single_responsibility.get_prob(item['docstring']) >= 0.5:
                    final_result_dict[py_file_path][func_name].append({'name': f"함수 {item['name']} 단일 책임 원칙 위반",
                                                                       'type': 'docstring',
                                                                       'line': line_no})

                if text_embedding_model_docstring_and_name.get_similarity(item['name'], item['docstring']) >= 0.5:
                    final_result_dict[py_file_path][func_name].append({'name': f"함수 {item['name']} - docstring 불일치",
                                                                       'type': 'docstring',
                                                                       'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_commented_codes(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model = self.text_embedding_models.get('default')

        final_result_dict = defaultdict(dict)

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)
            py_code_lines = py_code.split('\n')
            comments = []
            current_comment = ''

            for line_idx, line in enumerate(py_code_lines):
                comment = extract_comment(line)

                if comment == line and comment:
                    current_comment += comment[1:].strip() + ' '
                else:
                    if current_comment:
                        comments.append({'line': line_idx, 'comment': current_comment})
                    current_comment = ''
                    if comment:
                        comments.append({'line': line_idx + 1, 'comment': comment[1:]})

                if line_idx == len(py_code_lines) - 1 and current_comment:
                    comments.append({'line': line_idx, 'comment': current_comment})

            for comment in comments:
                if text_embedding_model.get_prob(comment) >= 0.5:
                    line_no = comment['line']
                    func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]

                    final_result_dict[py_file_path][func_name].append({'name': comment['comment'],
                                                                       'type': 'comment',
                                                                       'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_empty_file(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)
            py_code_lines = py_code.split('\n')
            is_empty = True

            for line_idx, line in enumerate(py_code_lines):
                comment = extract_comment(line)
                if comment != line:
                    is_empty = False
                    break

            if is_empty and 'TODO' not in py_code.upper():
                final_result_dict[py_file_path][''].append({'name': '파일이 비어 있지만 TODO 코멘트가 없습니다.',
                                                            'type': 'empty file',
                                                            'line': 1})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def run_code_review(self) -> dict[str, str]:
        checks = [
            'unused',
            'unnecessary_prints',
            'duplicates',
            'similar_variables',
            'same_func_args',
            'names',
            'return_matched_with_func_name',
            'library_orders',
            'func_docstring',
            'commented_codes',
            'empty_file'
        ]

        return {
            f'01_{name}': getattr(self, f'_check_{name}')()
            for name in checks
        }


class PythonBasicConventionChecker(DefaultCodeChecker):

    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()
        self._get_function_name_by_line()

    def _check_const(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model = self.text_embedding_models.get('default')

        final_result_dict = defaultdict(dict)

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            matched_lines = check_regex_matched_lines(py_code, r'(".*?"|\'.*?\'|\b\d+(?:\.\d+)?\b)')
            for line in matched_lines:
                if text_embedding_model.get_prob(line) >= 0.5:
                    line_no = line['line_no']
                    func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]

                    final_result_dict[py_file_path][func_name].append({'name': ellipse_str(line['line'].strip()),
                                                                       'type': 'const value',
                                                                       'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_line_length(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)
            lines = py_code.split('\n')

            for line_idx, line in enumerate(lines):
                line_no = line_idx + 1
                func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]

                if len(line) > self.max_line_length:
                    final_result_dict[py_file_path][func_name].append(
                        {'name': f'{ellipse_str(line.strip())} with length {len(line)}',
                         'type': 'const value',
                         'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_files(self) -> str:
        code_path_str = '(코드 전체 경로)'
        final_result_dict = defaultdict(dict)
        final_result_dict[code_path_str] = defaultdict(list)

        required_files = ['README.md', 'pyproject.toml']

        for required_file in required_files:
            if not os.path.exists(os.path.join(self.code_path, required_file)):
                if required_file == 'pyproject.toml' and len(self.py_codes.keys()) < 3:
                    continue

                final_result_dict[code_path_str][code_path_str].append({'name': f'오류: {required_file} 파일이 없습니다.',
                                                                        'type': 'no_file',
                                                                        'line': 0})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_functions(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            func_defs = [item for item in parsed_py_code if item['type_name'] == 'function_def']

            for item in func_defs:
                line_no = item['line']
                start_line = item['info']['start_line']
                end_line = item['info']['end_line']
                func_name = item['info']['name']
                func_length = end_line - start_line + 1

                if func_length > self.max_func_lines:
                    final_result_dict[py_file_path][func_name].append(
                        {'name': f'{func_name} 너무 긺 ({func_length} 줄 > {self.max_func_lines} 줄)',
                         'type': 'too_long_function',
                         'line': line_no})

                if func_length >= 10 and not item.get('docstring', None):
                    final_result_dict[py_file_path][func_name].append(
                        {'name': f'{func_name} docstring 없음',
                         'type': 'no_docstring',
                         'line': line_no})

                annotations = item['info']['args'].get('annot', None)
                if annotations and None in annotations:
                    final_result_dict[py_file_path][func_name].append(
                        {'name': f'{func_name} 의 일부 또는 전체 인수에 type hint 없음',
                         'type': 'no_type_hint_args',
                         'line': line_no})

                return_type = item['info'].get('return_type', None)
                if return_type is None:
                    final_result_dict[py_file_path][func_name].append(
                        {'name': f'{func_name} 의 return 값에 type hint 없음',
                         'type': 'no_type_hint_return',
                         'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_indent(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)
            lines = py_code.split('\n')
            current_base_indent = 0
            current_depth = 0

            for line_idx, line in enumerate(lines):
                line_no = line_idx + 1
                current_indent = int((len(line) - len(line.lstrip())) / self.code_indent)

                if line.strip().startswith('def '):
                    current_base_indent = current_indent + 1
                    current_depth = 0

                elif current_indent < current_base_indent and is_valid_code(line):
                    current_base_indent = current_indent
                    current_depth = 0

                if current_indent < current_depth and is_valid_code(line):
                    current_depth = current_indent

                if len(line) - len(line.lstrip()) == (current_depth + 1) * self.code_indent:
                    current_depth += 1

                if current_depth - current_base_indent >= 4 and is_valid_code(line):
                    func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                    final_result_dict[py_file_path][func_name].append({'name': ellipse_str(line.strip()),
                                                                       'type': 'too much indent',
                                                                       'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def run_code_review(self) -> dict[str, str]:
        checks = [
            'const',
            'line_length',
            'files',
            'functions',
            'indent'
        ]

        return {
            f'02_{name}': getattr(self, f'_check_{name}')()
            for name in checks
        }


class PythonSimplificationChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()
        self._get_function_name_by_line()

    def _init_final_result_dict(self) -> None:
        self.final_result_dict = defaultdict(dict)
        for py_file_path, py_code in self.py_codes.items():
            self.final_result_dict[py_file_path] = defaultdict(list)

    def _check_suggest_list_comprehension(self) -> str:
        self._init_final_result_dict()

        self._add_regex_matched_lines(
            regex=r"for\s+(\w+)\s+in\s+([^\n:]+):\s*\n\s*(\w+)\.append\(([^)]+)\)",
            match_func=lambda x: check_a_in_b(a=x[0], b=x[3]))

        self._add_regex_matched_lines(
            regex=r"for\s+(\w+)\s+in\s+([^\n:]+):\s*\n\s*if\s+(.*?):\s*\n\s*(\w+)\.append\(([^)]+)\)",
            match_func=lambda x: check_a_in_b(a=x[0], b=x[2]) and check_a_in_b(a=x[0], b=x[4]))

        self._add_regex_matched_lines(
            regex=(rf'(\w+)\s*=\s*[{QUOTES}][{QUOTES}]\s*\n' +
                   r'\s*for\s+(\w+)\s+in\s+([^\n:]+):\s*\n\s*(\w+)\s*\+=\s*([^\n]+)'),
            match_func=lambda x: x[0] == x[3] and check_a_in_b(a=x[1], b=x[4]))

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_generator_expression(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r'.*\b(sum|max|min|all|any|set)\s*\(\s*\[\s*(.+?\bfor\b.+?)\s*\]\s*\)')

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_if_to_dict(self) -> str:
        final_result_dict = defaultdict(dict)
        str_pattern_1 = '"[^"]*"'
        str_pattern_2 = "'[^']*'"

        regex = (r"^\s*(if|elif)\b\s*\(?\s*([a-zA-Z_]\w*)\s*" +
                 rf"(<=|>=|==)\s*([a-zA-Z_]\w*|\d+(?:\.\d+)?|{str_pattern_1}|{str_pattern_2})\s*\)?\s*:")

        def update_final_result_dict(py_file_path: str, line_no: int, current_if_elifs: list[dict]):
            if (len(current_if_elifs) >= 3
                    and len(set(info['var_name'] for info in current_if_elifs)) == 1
                    and len(set(info['simplified_body'] for info in current_if_elifs)) == 1):

                func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                final_result_dict[py_file_path][func_name].append({'name': 'if-elif-elif-else 패턴',
                                                                   'type': 'if-elif-elif-else',
                                                                   'line': line_no})

            current_if_elifs.clear()

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)
            lines = py_code.split('\n')

            last_if_elif_line_idx = -1
            current_if_elifs = []

            for line_idx, line in enumerate(lines):
                line_no = line_idx + 1
                matched = re.match(regex, line)

                if matched:
                    matched_groups = list(matched.groups())
                    keyword = matched_groups[0]
                    var_name = matched_groups[1]

                    if keyword in ['if', 'elif']:
                        last_if_elif_line_idx = line_idx

                    current_if_elifs.append({'keyword': keyword, 'var_name': var_name})

                elif line_idx - last_if_elif_line_idx >= 2:
                    update_final_result_dict(py_file_path, line_no, current_if_elifs)

                elif line_idx - last_if_elif_line_idx == 1:
                    if len(current_if_elifs) >= 1:
                        current_if_elifs[-1]['simplified_body'] = simplify_code(line)

                if line_idx == len(lines) - 1:
                    update_final_result_dict(py_file_path, line_no, current_if_elifs)

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_path_format(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=rf'.*([{QUOTES}])(?:[a-zA-Z]:)?[/\\]*(?:[^/\\\r\n]+[/\\]+)+[^/\\\r\n]+\.[a-zA-Z0-9]+([{QUOTES}])',
            forward_window_size=1)

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_defaultdict(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, py_code in self.py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            definition_lines = check_regex_matched_lines(
                py_code, r"([a-zA-Z_]\w*)\s*=\s*\{\}")
            value_definition_lines = check_regex_matched_lines(
                py_code, rf"([a-zA-Z_]\w*)\s*\[.*?\]\s*=\s*(?:\[\]|0|''|{TWO_DOUBLE_QUOTES})")

            for def_line, val_line in product(definition_lines, value_definition_lines):
                val_line_no = val_line['line_no']
                def_line_no = def_line['line_no']
                val_line_ = val_line['line']
                def_line_ = def_line['line']

                if val_line_no > def_line_no:
                    if def_line_.split('=')[0].strip() == val_line_.split('[')[0].strip():
                        func_name = self.function_name_by_line_for_codebase[py_file_path][def_line_no]
                        final_result_dict[py_file_path][func_name].append({'name': ellipse_str(val_line_.strip()),
                                                                           'type': 'should use defaultdict',
                                                                           'line': def_line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_any_all(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(False|True)\s+' +
                   r'for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s+' +
                   r'if(\s+[^:]+):\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(True|False)\s+'),
            match_func=lambda x: len(x) >= 7 and x[0] == x[5] and x[1] != x[6] and x[2] in x[4],
            forward_window_size=7)

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_zip(self) -> str:
        pass

    def _check_enumerate(self) -> str:
        pass

    def _check_itertools_product(self) -> str:
        pass

    def _check_just_read_write_to_read_write_text(self) -> str:
        pass

    def _check_sentence_empty(self) -> str:
        pass

    def _check_handle_none(self) -> str:
        pass

    def _check_extend(self) -> str:
        pass

    def _check_count(self) -> str:
        pass

    def _check_index(self) -> str:
        pass

    def _check_str_join(self) -> str:
        pass

    def _check_use_get(self) -> str:
        pass

    def _check_use_map(self) -> str:
        pass

    def run_code_review(self) -> dict[str, str]:
        checks = [
            'suggest_list_comprehension',
            'generator_expression',
            'if_to_dict',
            'path_format',
            'defaultdict',
            'any_all',
            'zip',
            'enumerate',
            'itertools_product',
            'just_read_write_to_read_write_text',
            'sentence_empty',
            'handle_none',
            'extend',
            'count',
            'index',
            'str_join',
            'use_get',
            'use_map',
        ]

        print(self._check_any_all())

        return {
            f'03_{name}': getattr(self, f'_check_{name}')()
            for name in checks
        }


class PythonOtherPythonicChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()


class PythonExceptionsChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()


class PythonCohesivenessAndClassChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()


class PyTorchChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()


class EntireCodeChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)

        self.python_basics_checker = PythonBasicsChecker(py_codes, config, code_path)
        self.python_basic_convention_checker = PythonBasicConventionChecker(py_codes, config, code_path)
        self.python_simplification_checker = PythonSimplificationChecker(py_codes, config, code_path)
        self.python_other_pythonic_checker = PythonOtherPythonicChecker(py_codes, config, code_path)
        self.python_exceptions_checker = PythonExceptionsChecker(py_codes, config, code_path)
        self.python_cohesiveness_and_class_checker = PythonCohesivenessAndClassChecker(py_codes, config, code_path)
        self.pytorch_checker = PyTorchChecker(py_codes, config, code_path)

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


def default_code_review_func(py_codes: dict[str, str], config: dict, code_path: str) -> dict[str, str]:
    """Default code review function for Oh-LoRA 👱‍♀️ Code Assistant."""

    default_code_checker = EntireCodeChecker(py_codes=py_codes, config=config, code_path=code_path)
    return default_code_checker.run_code_review()
