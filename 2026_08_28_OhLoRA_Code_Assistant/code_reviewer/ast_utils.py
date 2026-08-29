import ast
from pathlib import Path

from collections import defaultdict
from operator import itemgetter


TYPE_TO_NAME = {
    ast.ClassDef: 'class',
    ast.Import: 'import',
    ast.ImportFrom: 'import_from',
    ast.FunctionDef: 'function_def',
    ast.Call: 'call',
    ast.Name: 'name',
    ast.Constant: 'constant',
    ast.Attribute: 'attribute'
}


def find_unused_variables(source_code):
    tree = ast.parse(source_code)
    parse_results = []

    for node in ast.walk(tree):
        line_no = getattr(node, 'lineno', None)
        col_offset = getattr(node, 'col_offset', None)
        if line_no is None or col_offset is None:
            continue

        node_type = type(node)
        type_name = TYPE_TO_NAME.get(node_type, None)
        if type_name is None:
            print(node_type, type_name)
            continue

        parse_result: dict = {'line': line_no, 'col': col_offset, 'type_name': type_name}

        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            parse_result['info'] = {'name': node.name, 'bases': bases}

        elif isinstance(node, ast.Import):
            import_names = [{'name': alias.name, 'as_name': alias.asname} if alias.asname else {'name': alias.name}
                            for alias in node.names]
            parse_result['info'] = {'import_names': import_names}

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            import_names = [{'name': alias.name, 'as_name': alias.asname} if alias.asname else {'name': alias.name}
                            for alias in node.names]
            parse_result['info'] = {'mod': mod, 'import_names': import_names}

        elif isinstance(node, ast.FunctionDef):
            args_info = defaultdict(list)

            args = node.args.args
            defaults = node.args.defaults
            padded_defaults = [None] * (len(args) - len(defaults)) + list(defaults)

            for arg, default in zip(args, padded_defaults):
                args_info['name'].append(arg.arg)
                args_info['annot'].append(ast.unparse(arg.annotation) if arg.annotation else None)
                args_info['default'].append(default.value if default else None)
            parse_result['info'] = {'name': node.name, 'args': dict(args_info)}

        elif isinstance(node, ast.Call):
            func_name = ast.unparse(node.func)
            args = [ast.unparse(a) for a in node.args]
            kwargs = [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]  # 키워드 인수 값
            parse_result['info'] = {'func_name': func_name, 'args': args, 'kwargs': kwargs}

        elif isinstance(node, ast.Name):
            parse_result['info'] = {'name': node.id}

        elif isinstance(node, ast.Constant):
            parse_result['info'] = {'value': node.value, 'type': type(node.value)}

        elif isinstance(node, ast.Attribute):
            parse_result['info'] = {'attr': node.attr}

        parse_results.append(parse_result)

    parse_results.sort(key=itemgetter('col'))
    parse_results.sort(key=itemgetter('line'))

    for parse_result in parse_results:
        print(parse_result)


if __name__ == '__main__':
    py_code = Path("D:/AI_Projects/2026_08_28_OhLoRA_Code_Assistant/code_reviewer/code_reviewer.py").read_text(encoding='utf-8')
    find_unused_variables(py_code)

