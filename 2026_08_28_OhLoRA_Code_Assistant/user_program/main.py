
import os
import sys
import argparse

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from backend.entire_process import run_entire_process


"""
Run these test cases below at '2026_08_28_OhLoRA_Code_Assistant/user_program' directory.

Test Case        : python main.py --code_path ../code_reviewer/test_cases --verbose True
Test Case Result : user_program/test_case_result.txt

Test Case        : python main.py --code_path ../code_reviewer/test_cases --except_path ../code_reviewer/test_cases/path_test --verbose True
Test Case Result : user_program/test_case_result_with_except_path.txt

Test Case        : python main.py --code_path ../code_reviewer/test_cases/tc02.py --verbose True
Test Case Result : user_program/test_case_result_single_file.txt

Test Case        : python main.py --code_path "../code_reviewer/test_cases/tc02.py ||| ../code_reviewer/test_cases/tc03.py" --verbose True
Test Case Result : user_program/test_case_result_multiple_files.txt
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
