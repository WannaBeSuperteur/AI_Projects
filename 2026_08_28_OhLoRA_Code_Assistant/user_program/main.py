
import os
import sys
import argparse

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from backend.entire_process import run_entire_process


"""
Test Case        : python main.py --code_path ../code_reviewer/test_cases --verbose True
Test Case Result : user_program/test_case_result.txt
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_path",
                        required=True,
                        help="Path of Python code (or codebase, directory, repo) to review.")
    parser.add_argument("--except_path",
                        required=False,
                        default='N/A',
                        help="When reviewing Python codebase, exclude this directory path. ('N/A' if no except path)")
    parser.add_argument("--verbose",
                        type=bool,
                        required=False,
                        default=False)

    args = parser.parse_args()
    print(f'parsed args : {args}')

    entire_process_result = run_entire_process(code_path=args.code_path,
                                               except_path=None if args.except_path == 'N/A' else args.except_path,
                                               verbose=args.verbose)

    print('==== CODE REVIEW RESULT ====')
    print(entire_process_result)
