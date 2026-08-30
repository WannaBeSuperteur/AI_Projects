import ast

from collections import defaultdict
from operator import itemgetter


TYPE_TO_NAME = {
    ast.Import: 'import',
    ast.ImportFrom: 'import_from',
    ast.ClassDef: 'class',
    ast.FunctionDef: 'function_def',
    ast.AsyncFunctionDef: 'async_function_def',
    ast.If: 'if',
    ast.Call: 'call',
    ast.Name: 'name',
    ast.Constant: 'constant',
    ast.Attribute: 'attribute'
}


def parse_function_def(node: ast.AST) -> dict:
    args_info = defaultdict(list)

    args = node.args.args
    defaults = node.args.defaults
    padded_defaults = [None] * (len(args) - len(defaults)) + list(defaults)

    for arg, default in zip(args, padded_defaults):
        args_info['name'].append(arg.arg)
        args_info['annot'].append(ast.unparse(arg.annotation) if arg.annotation else None)
        args_info['default'].append(getattr(default, 'value', None) if default else None)

    return {
        'name': node.name,
        'args': dict(args_info),
        'start_line': getattr(node, 'lineno', None),
        'end_line': getattr(node, 'end_lineno', None)
    }


def parse_py_code(source_code: str, verbose: bool = False) -> list[dict]:
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

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parse_result['info'] = parse_function_def(node)

        elif isinstance(node, ast.Call):
            func_name = ast.unparse(node.func)
            args = [ast.unparse(a) for a in node.args]
            kwargs = [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]  # 키워드 인수 값
            parse_result['info'] = {'func_name': func_name, 'args': args, 'kwargs': kwargs}

        elif isinstance(node, ast.Name):
            ctx_type = type(node.ctx).__name__
            parse_result['info'] = {'name': node.id, 'ctx': ctx_type}

        elif isinstance(node, ast.Constant):
            parse_result['info'] = {'value': node.value, 'type': type(node.value)}

        elif isinstance(node, ast.Attribute):
            parse_result['info'] = {'attr': node.attr}

        elif isinstance(node, ast.If):
            parse_result['if'] = {'test': ast.unparse(node.test)}

        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if docstring := ast.get_docstring(node):
                parse_result['docstring'] = docstring

        parse_results.append(parse_result)

    parse_results.sort(key=itemgetter('col'))
    parse_results.sort(key=itemgetter('line'))

    if verbose:
        for parse_result in parse_results:
            print(parse_result)

    return parse_results

