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
from itertools import chain, product, groupby
from ast_utils import parse_py_code

PRESERVED_WORDS = set(keyword.kwlist) | set(dir(builtins))

QUOTES = "'" + '"'
TWO_DOUBLE_QUOTES = '""'
QUOTES_BOUND = rf"[{QUOTES}].*?[{QUOTES}]"
ANY_CONST_OR_VAR = rf"({QUOTES_BOUND}|[\w.]+)"


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

    def _add_regex_matched_lines(self,
                                 regex: str,
                                 match_func: Optional[Callable] = None,
                                 file_line_match_func: Optional[Callable] = None,
                                 type_name: str = 'regex',
                                 forward_lines: int = 5) -> None:

        for py_file_path, py_code in self.py_codes.items():
            lines = [line.strip() for line in py_code.split('\n')]
            lines = [line[:len(line)-len(extract_comment(line))] for line in lines]

            line_idx = 0
            while line_idx < len(lines):
                line_no = line_idx + 1
                lines_to_search = '\n'.join(lines[line_idx:line_idx+forward_lines])
                matched = re.match(regex, lines_to_search)

                if matched:
                    matched_text = matched.group(0)
                    matched_groups = list(matched.groups())

                    func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]
                    lines_to_show = lines_to_search.replace('\n', ' ')
                    if len(lines_to_show) > 40:
                        lines_to_show = lines_to_show[:36] + ' ...'

                    is_matched = match_func is None or match_func(matched_groups)
                    is_file_line_matched = file_line_match_func is None or file_line_match_func(matched_groups,
                                                                                                py_file_path,
                                                                                                line_no)

                    if is_matched and is_file_line_matched:
                        self.final_result_dict[py_file_path][func_name].append({'name': lines_to_show,
                                                                                'type': type_name,
                                                                                'line': line_no})
                        line_idx += matched_text.count('\n') + 1
                    else:
                        line_idx += 1
                else:
                    line_idx += 1

    def _init_final_result_dict(self) -> None:
        self.final_result_dict = defaultdict(dict)
        for py_file_path, py_code in self.py_codes.items():
            self.final_result_dict[py_file_path] = defaultdict(list)

    def _find_if_elif_else_patterns(self, additional_check_func: Optional[Callable] = None) -> defaultdict:
        final_result_dict = defaultdict(dict)
        str_pattern_1 = '"[^"]*"'
        str_pattern_2 = "'[^']*'"

        regex = (r"^\s*(if|elif)\b\s*\(?\s*([a-zA-Z_](\w|\.)*)\s*" +
                 rf"(<=|>=|==)\s*([a-zA-Z_]\w*|\d+(?:\.\d+)?|{str_pattern_1}|{str_pattern_2})\s*\)?\s*:")

        def update_final_result_dict(py_file_path: str, line_no: int, current_if_elifs: list[dict],
                                     additional_check_func: Optional[Callable] = None):

            if (len(current_if_elifs) >= 3
                    and len(set(info['var_name'] for info in current_if_elifs)) == 1
                    and len(set(info['simplified_body'] for info in current_if_elifs)) == 1
                    and (additional_check_func is None or additional_check_func(current_if_elifs))):
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
                    update_final_result_dict(py_file_path, line_no, current_if_elifs, additional_check_func)

                elif line_idx - last_if_elif_line_idx == 1:
                    if len(current_if_elifs) >= 1:
                        current_if_elifs[-1]['simplified_body'] = simplify_code(line)

                if line_idx == len(lines) - 1:
                    update_final_result_dict(py_file_path, line_no, current_if_elifs, additional_check_func)

        return final_result_dict

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
        self.final_result_dict = self._find_if_elif_else_patterns()
        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_path_format(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=rf'.*([{QUOTES}])(?:[a-zA-Z]:)?[/\\]*(?:[^/\\\r\n]+[/\\]+)+[^/\\\r\n]+\.[a-zA-Z0-9]+([{QUOTES}])',
            forward_lines=1)

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
                   r'if(\s+[^:]+):\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(True|False)'),
            match_func=lambda x: len(x) >= 7 and x[0] == x[5] and x[1] != x[6] and x[2] in x[4],
            forward_lines=7)

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_zip(self) -> str:
        def _match_zip(matched: list[str]) -> bool:
            idx_matched = matched[0] == matched[3] and matched[3] == matched[5]
            name_matched = matched[1] == f'len({matched[2]})' or matched[1] == f'len({matched[4]})'
            return idx_matched and name_matched

        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=(r"for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+range\s*\(\s*(len\s*\([a-zA-Z_][a-zA-Z0-9_]*\))\s*\)\s*:" +
                   2 * r"[\s\S]*?([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]"),
            match_func=_match_zip,
            forward_lines=15)

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_enumerate(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"\bfor\s+\w+\s+in\s+range\s*\(\s*len\s*\([^)]+\)\s*\)\s*:")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_itertools_product(self) -> str:
        self._init_final_result_dict()

        regex_for_in_range = r'for\s+\w+\s+in\s+range\s*\((.*?)\)\s*:'
        self._add_regex_matched_lines(regex=rf"\b{regex_for_in_range}\s+{regex_for_in_range}")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_just_read_write_to_read_write_text(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=(r"with\s+open\s*\((.*?)\)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s+" +
                   r"(?:(?:([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\2\.read\s*\((\s*)\))|(?:\2\.write\s*\((.*?)\)))"))

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_sentence_empty(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"if\s+(?:not\s+)?len\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*:")
        self._add_regex_matched_lines(regex=r"if\s+(?:not\s+)?len\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*==\s*0\s*:")
        self._add_regex_matched_lines(regex=rf"if\s+(?:not\s+)?([a-zA-Z_]\w*)\s*==\s*[{QUOTES}][{QUOTES}]\s*:")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_handle_none(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=rf".*?{ANY_CONST_OR_VAR}\s+in\s+([\w.]+)\s+and\s+\2\s*\[\s*\1\s*\]")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_extend(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=r".*?for\s+([\w.]+)\s+in\s+([\w.]+)\s*:\s*\n\s*([\w.]+)\.append\s*\(\s*\1\s*\)")

        self._add_regex_matched_lines(
            regex=r".*?([\w.]+)\s*\+=\s*\[\s*(.*?)\s+for\s+([\w.]+)\s+in\s+([\w.]+)\s*\]")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_count(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=(r".*?([\w.]+)\s*=\s*0\s*\n\s*for\s+([\w.]+)\s+in\s+([\w.]+)\s*:" +
                   rf"\s*\n\s*if\s+\2\s*==\s*{ANY_CONST_OR_VAR}\s*:\s*\n\s*\1\s*\+=\s*1"))

        self._add_regex_matched_lines(
            regex=(r".*?([\w.]+)\s*=\s*len\s*\(\s*\[\s*([\w.]+)\s+" +
                   rf"for\s+\2\s+in\s+([\w.]+)\s+if\s+\2\s*==\s*{ANY_CONST_OR_VAR}\s*\]\s*\)"))

        self._add_regex_matched_lines(
            regex=(rf".*?len\s*\(\s*list\s*\(\s*filter\s*\(\s*lambda\s+([\w.]+)\s*:" +
                   rf"\s*\1\s*==\s*{ANY_CONST_OR_VAR}\s*,\s*([\w.]+)\s*\)\s*\)?\s*\)"))

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_index(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=(r".*?([\w.]+)\s*=\s*([-\w.]+|[\w.]+)\s*\n" +
                   rf"\s*for\s+([\w.]+)\s+in\s+range\s*\(\s*len\s*\(\s*{ANY_CONST_OR_VAR}\s*\)\s*\)\s*:\s*\n" +
                   rf"\s*if\s+\4\s*\[\s*\3\s*\]\s*==\s*{ANY_CONST_OR_VAR}\s*:\s*\n" +
                   r"\s*([\w.]+)\s*=\s*\3\s*\n\s*break"),
            forward_lines=10)

        self._add_regex_matched_lines(
            regex=(r".*?([\w.]+)\s*=\s*next\s*\(\s*([\w.]+)\s+for\s+\2\s*,\s*([\w.]+)\s+" +
                   rf"in\s+enumerate\s*\(\s*([\w.]+)\s*\)\s+if\s+\3\s*==\s*{ANY_CONST_OR_VAR}"))

        self._add_regex_matched_lines(
            regex=(r".*?([\w.]+)\s*=\s*([\w.]+)\s*\n" +
                   r"\s*while\s+\1\s*<\s*len\s*\(\s*([\w.]+)\s*\)\s*:\s*\n" +
                   rf"\s*if\s+\3\s*\[\s*\1\s*\]\s*==\s*{ANY_CONST_OR_VAR}\s*:\s*\n" +
                   r"\s*break\s*\n\s*\1\s*\+=\s*1"),
            forward_lines=10)

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_str_join(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=fr"([\w.]+)\s*=\s*(''|{TWO_DOUBLE_QUOTES})\s*(\n|\n\s*\n)\s*for.*?:\n\1\s*\+=")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_use_map(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=r".*?(sum|max|min)\s*\(\s*([\w.]+)\s*\(\s*([\w.]+)\s*\)\s+for\s+\3\s+in[^\n]*\)")

        return convert_to_human_friendly_review(self.final_result_dict)

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
            'use_map',
        ]

        return {
            f'03_{name}': getattr(self, f'_check_{name}')()
            for name in checks
        }


class PythonOtherPythonicChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()
        self._get_function_name_by_line()

    def _check_unpacking(self) -> str:
        value_assign = rf"([\w.]+)\s*=\s*([\w.]+)\s*\[({QUOTES_BOUND}|[\w.]+|[\w.]+:)]"

        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=rf"{value_assign}\s*\n\s*{value_assign}",
            match_func=lambda x: x[1] == x[4])

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_open_file(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=rf"([\w.]+)\s*=\s*open\s*\(\s*{ANY_CONST_OR_VAR}\s*,\s*{ANY_CONST_OR_VAR}\s*\)")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_key_itemgetter(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=rf"([\w.]+)\s*\.\s*sort\s*\(\s*key\s*=\s*lambda\s+([\w.]+)\s*:\s*\2\s*\[{ANY_CONST_OR_VAR}\]\)")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_f_string(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=rf"^..*?({QUOTES_BOUND}\s*\+\s*([\w.]+)|([\w.]+)\s*\+\s*{QUOTES_BOUND})")

        self._add_regex_matched_lines(
            regex=rf"^..*?({QUOTES_BOUND}\s*\+\s*\(.*?\)|\(.*?\)\s*\+\s*{QUOTES_BOUND})")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_collections_itertools_glob(self) -> str:
        self._init_final_result_dict()

        # 1. collections
        self._add_regex_matched_lines(
            regex=(r"for\s+([\w.]+)\s+in\s+([\w.]+)\s*:\s*\n" +
                   r"\s*if\s+\1\s+in\s+([\w.]+)\s*:\s*\n" +
                   r"\s*\3\s*\[\s*\1\s*\]\s*\+=\s*1\s*\n" +
                   r"\s*else\s*:\s*\n" +
                   r"\s*\3\s*\[\s*\1\s*\]\s*=\s*1"),
            forward_lines=10)

        self._add_regex_matched_lines(
            regex=(r"\s*for\s+([\w.]+)\s+in\s+([\w.]+)\s*:\s*\n" +
                   r"\s*([\w.]+)\s*\[\s*\1\s*\]\s*=\s*\3\s*\.\s*get\s*\(\s*\1\s*,\s*0\s*\)\s*\+\s*1"))

        # 2. itertools.chain
        self._add_regex_matched_lines(
            regex=(r"\s*for\s+([\w.]+)\s+in\s+([\w.]+)\s*:\s*\n" +
                   r"\s*for\s+([\w.]+)\s+in\s+\1\s*:\s*\n" +
                   r".*?.\s*append\s*\(\s*\3\s*\)"))

        self._add_regex_matched_lines(
            regex=rf"^..*?({ANY_CONST_OR_VAR}\s*\+\s*list\s*\(.*?\)|list\s*\(.*?\)\s*\+\s*{ANY_CONST_OR_VAR})")

        # 3. glob
        self._add_regex_matched_lines(
            regex=(r"for\s+([\w.]+)\s+in\s+os\s*\.\s*listdir\s*\(\s*[\w.]+\s*\)\s*:\s*\n" +
                   r"\s*if(\s+.*?\s*):\s*\n"),
            match_func=lambda x: x[0] in x[1])

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_func_args_bindable(self) -> str:
        final_result_dict = defaultdict(dict)

        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model_bindable = self.text_embedding_models.get('default')
        text_embedding_model_dynamic = self.text_embedding_models.get('default')

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)
            function_defs = [item for item in parsed_py_code if item['type_name'] == 'function_def']

            for item in function_defs:
                line_no = item['line']
                func_name = item['info']['name']
                arg_names = item['info'].get('args', {}).get('name', None)

                if arg_names is not None:
                    arg_name_list = ','.join(arg_names)

                    if text_embedding_model_bindable.get_prob(arg_name_list) >= 0.5:
                        final_result_dict[py_file_path][func_name].append(
                            {'name': f'{func_name}({ellipse_str(arg_name_list)})',
                             'type': 'bindable_args',
                             'line': line_no})

                    if text_embedding_model_dynamic.get_prob(arg_name_list) >= 0.5:
                        final_result_dict[py_file_path][func_name].append(
                            {'name': f'{func_name}({ellipse_str(arg_name_list)})',
                             'type': 'dynamic_args',
                             'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_attribute_getattr(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=(rf"if\s+hasattr\s*\(\s*([\w.]+)\s*,\s*{ANY_CONST_OR_VAR}\s*\)\s*:\s*\n" +
                   r"\s*([\w.]+)\s*=\s+(.*)\n\s*else\s*:\s*\n" +
                   r"\s*([\w.]+)\s*=\s+"),
            match_func=lambda x: x[2] == x[4])

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_regex_r(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(
            regex=(r"^..*?re\s*\.\s*(sub|match|search|compile|findall|finditer|split|fullmatch)\s*" +
                   fr"\(\s*{QUOTES_BOUND}.*\)"))

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_func_lambda(self) -> str:
        stored_name_dict = defaultdict(dict)

        def flmf(matched_groups, py_file_path, line_no):
            func_name = matched_groups[0]
            return func_name in stored_name_dict[py_file_path][line_no]

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            stored_name_dict[py_file_path] = defaultdict(set)

            stored_names = [item for item in parsed_py_code
                            if item['type_name'] == 'name' and item['info'].get('ctx', None) == 'Store']

            for item in stored_names:
                stored_name_dict[py_file_path][item['line']].add(item['info']['name'])

        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"([\w.]+)\s*=\s*lambda\s+([\w.|\s*,\s*]+)\s*:\s*",
                                      file_line_match_func=flmf)

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_prefix_suffix(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"^..*?([\w.]+)\s*\[(:[0-9]+|-[0-9]+:)\]\s*==")

        return convert_to_human_friendly_review(self.final_result_dict)

    def run_code_review(self) -> dict[str, str]:
        checks = [
            'unpacking',
            'open_file',
            'key_itemgetter',
            'f_string',
            'collections_itertools_glob',
            'func_args_bindable',
            'attribute_getattr',
            'regex_r',
            'func_lambda',
            'prefix_suffix'
        ]

        return {
            f'04_{name}': getattr(self, f'_check_{name}')()
            for name in checks
        }


class PythonExceptionsChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()
        self._get_function_name_by_line()

    def _check_exception_ignored(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"except\s*:\s*\n\s*pass")
        self._add_regex_matched_lines(regex=r"except\s+(BaseException|Exception)\s*:\s*\n\s*pass")
        self._add_regex_matched_lines(regex=r"except\s+(BaseException|Exception)\s+as\s+([\w.]+):\s*\n\s*pass")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_exception_type(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"except\s*:\s*\n")
        self._add_regex_matched_lines(regex=r"except\s+(BaseException|Exception)\s*:\s*\n")
        self._add_regex_matched_lines(regex=r"except\s+(BaseException|Exception)\s+as\s+([\w.]+):\s*\n")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_func_arg_error_prevent(self) -> str:
        stored_arg_name_dict = defaultdict(dict)

        def flmf(matched_groups, py_file_path, line_no):
            arg_name = matched_groups[0]
            return arg_name in stored_arg_name_dict[py_file_path][line_no]

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            stored_arg_name_dict[py_file_path] = defaultdict(set)
            function_defs = [item for item in parsed_py_code if item['type_name'] == 'function_def']

            for item in function_defs:
                arg_names = item['info'].get('args', {}).get('name', None)

                if arg_names is not None:
                    start_line = item['line']
                    def_end_line = item['info']['end_line'] - item['info']['body'].count('\n') - 1

                    for line_no in range(start_line, def_end_line + 1):
                        stored_arg_name_dict[py_file_path][line_no].update(set(arg_names))

        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"^..*?\s*([\w.]+)\s*:\s*(dict|list)\s*=\s*(\{|\[).*(\}|\])\s*\)",
                                      file_line_match_func=flmf)

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_assertion_try_except(self) -> str:
        self._init_final_result_dict()
        self._add_regex_matched_lines(regex=r"except\s+AssertionError\s*(\s*as\s+([\w.]+)\s*:|:)")

        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_python_keywords_args(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            store_cases = [item for item in parsed_py_code if item.get('info', {}).get('ctx', None) == 'Store']
            store_cases_bad = [item for item in store_cases if item['info']['name'] in PRESERVED_WORDS]

            for item in store_cases_bad:
                line_no = item['line']
                func_name = self.function_name_by_line_for_codebase[py_file_path][line_no]

                final_result_dict[py_file_path][func_name].append({'name': f"변수 {item['info']['name']}",
                                                                   'type': 'bad_store_cases',
                                                                   'line': line_no})

            function_defs = [item for item in parsed_py_code if item['type_name'] == 'function_def']
            function_args = [{'line': item['line'],
                              'func_name': item['info']['name'],
                              'args': item['info'].get('args', {}).get('name', None)}
                             for item in function_defs]
            function_args_bad = [{'line': item['line'],
                                  'func_name': item['func_name'],
                                  'args': {x for x in item['args'] if x in PRESERVED_WORDS}}
                                 for item in function_args
                                 if item['args'] is not None]
            function_args_bad = [item for item in function_args_bad if isinstance(item['args'], set)]

            for item in function_args_bad:
                line_no = item['line']
                arg_names = item['args']
                func_name = item['func_name']

                for arg_name in arg_names:
                    final_result_dict[py_file_path][func_name].append({'name': f"함수 {func_name}의 인자 {arg_name}",
                                                                       'type': 'bad_func_args',
                                                                       'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def run_code_review(self) -> dict[str, str]:
        checks = [
            'exception_ignored',
            'exception_type',
            'func_arg_error_prevent',
            'assertion_try_except',
            'python_keywords_args'
        ]

        return {
            f'05_{name}': getattr(self, f'_check_{name}')()
            for name in checks
        }


class PythonCohesivenessAndClassChecker(DefaultCodeChecker):
    def __init__(self, py_codes: dict[str, str], config: dict, code_path: str):
        super().__init__(py_codes, config, code_path)
        self._parse_codes()
        self._get_function_name_by_line()

    def _check_refactor_into_class_case_1_same_args(self) -> str:
        final_result_dict = defaultdict(dict)

        for py_file_path, parsed_py_code in self.parsed_py_codes.items():
            final_result_dict[py_file_path] = defaultdict(list)

            function_defs = [item for item in parsed_py_code if item['type_name'] == 'function_def']
            function_and_args = [{'line': item['line'],
                                  'func_name': self.function_name_by_line_for_codebase[py_file_path][item['line'] - 1],
                                  'args_name': item['info'].get('args', {}).get('name', None)}
                                 for item in function_defs]
            function_and_args_ = groupby(function_and_args, key=itemgetter('func_name'))

            for func_name, items in function_and_args_:
                duplicate_count_except_first = 0
                arg_name_discovered = set()

                items_ = list(items)
                for item in items_:
                    if not item.get('args_name', None):
                        continue

                    arg_name_set = set(item['args_name'])
                    intersection_size = len(arg_name_discovered.intersection(arg_name_set))
                    arg_name_discovered.update(arg_name_set)
                    duplicate_count_except_first += max(intersection_size - 1, 0)

                if duplicate_count_except_first >= 4:
                    line_no = items_[0]['line']
                    final_result_dict[py_file_path][func_name].append({'name': f"중복된 함수 인수 너무 많음",
                                                                       'type': 'bad_func_args',
                                                                       'line': line_no})

        self.final_result_dict = final_result_dict
        return convert_to_human_friendly_review(final_result_dict)

    def _check_refactor_into_class_case_2_state_vars_if_else(self) -> str:
        if self.text_embedding_models.get('default') is None:
            return "no text embedding model"

        text_embedding_model = self.text_embedding_models.get('default')

        def check_is_state_value(info):
            text = f"if {info[0]['var_name']}: {info[0]['simplified_body']}"
            return text_embedding_model.get_prob(text) >= 0.5

        self.final_result_dict = self._find_if_elif_else_patterns(additional_check_func=check_is_state_value)
        return convert_to_human_friendly_review(self.final_result_dict)

    def _check_cohesion(self) -> str:
        pass

    def _check_prefix_for_only_in_class_methods(self) -> str:
        pass

    def run_code_review(self) -> dict[str, str]:
        checks = [
            'refactor_into_class_case_1_same_args',
            'refactor_into_class_case_2_state_vars_if_else',
            'cohesion',
            'prefix_for_only_in_class_methods'
        ]

        print(self._check_refactor_into_class_case_1_same_args())
        print(self._check_refactor_into_class_case_2_state_vars_if_else())

        return {
            f'06_{name}': getattr(self, f'_check_{name}')()
            for name in checks
        }


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
