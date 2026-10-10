
import os
import sys

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
CODE_REVIEWER_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'code_reviewer')
sys.path.append(CODE_REVIEWER_DIR_PATH)

from code_reviewer import run_entire_code_review


TEST_CASES_DIR = os.path.join(CODE_REVIEWER_DIR_PATH, 'test_cases')


def print_code_review_result(code_review_result: dict):
    for k, v in code_review_result.items():
        if k != 'item_counts':
            print(f'\n\n[ {k} ]')
            print(v)

    print('\n\n[ item_counts ]')
    for rule_id, rule_review_result in code_review_result['item_counts'].items():
        print(rule_id, rule_review_result)


def run_entire_process(code_path: str, verbose: bool = False, llm_final_comment: bool = True) -> dict:
    code_review_result = run_entire_code_review(code_path)
    if verbose:
        print_code_review_result(code_review_result)

    result_dict = {'top_items_eval_summary': code_review_result['top_items_eval_summary'],
                   'code_review_result_str': code_review_result['code_review_result_str'],
                   'mean_score': code_review_result['mean_score']}

    if llm_final_comment:
        pass  # TODO: LLM 생성 코드

    return result_dict


if __name__ == '__main__':
    entire_process_result = run_entire_process(code_path=TEST_CASES_DIR, verbose=True)

    for k, v in entire_process_result.items():
        print(f'\n\nFINAL [[ {k} ]]')
        print(str(v)[:1000])
