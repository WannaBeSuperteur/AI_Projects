
import os
import sys
import re
from pathlib import Path

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
CODE_REVIEWER_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'code_reviewer')
LLM_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'llm_model')

sys.path.append(PROJECT_DIR_PATH)
sys.path.append(CODE_REVIEWER_DIR_PATH)
sys.path.append(LLM_DIR_PATH)

from code_reviewer import run_entire_code_review
from run_inference import LLMInferenceEngine
from llm_model.utils import ANSWER_START_MARK, ANSWER_END_MARK


LLM_NAME = 'kanana-2-3b-instruct_fine_tuned'
TEST_CASES_DIR = os.path.join(CODE_REVIEWER_DIR_PATH, 'test_cases')
TEST_CASES_MULTIPLE_FILES = (f'{os.path.join(TEST_CASES_DIR, "tc02.py")} ||| ' +
                             f'{os.path.join(TEST_CASES_DIR, "tc03.py")} ||| ' +
                             f'{os.path.join(TEST_CASES_DIR, "tc04.py")} ||| ' +
                             f'{os.path.join(TEST_CASES_DIR, "tc05.py")} ||| ' +
                             f'{os.path.join(TEST_CASES_DIR, "tc06.py")} ||| ' +
                             f'{os.path.join(TEST_CASES_DIR, "404_new_era.txt")}')
LLM_FULL_MODEL_PATH = os.path.join(LLM_DIR_PATH, LLM_NAME, 'full_model')

FINAL_WITH_LLM = "FINAL (WITH LLM)"
FINAL_WO_LLM = "FINAL (W/O LLM)"

TOP_ITEMS_EVAL_SUMMARY = "top_items_eval_summary"
CODE_REVIEW_RESULT_STR = "code_review_result_str"
MEAN_SCORE = "mean_score"
LLM_ANSWER = "llm_answer"
LENGTH_TO_PRINT = 500


def print_code_review_result(code_review_result: dict):
    for k, v in code_review_result.items():
        if k != 'item_counts':
            print(f'\n\n[ {k} ]')
            print(v)

    print('\n\n[ item_counts ]')
    for rule_id, rule_review_result in code_review_result['item_counts'].items():
        print(rule_id, rule_review_result)


def generate_llm_comment(top_items_eval_summary: str) -> str:
    """Generate LLM answer using LLM inference engine."""

    full_prompt = f'{top_items_eval_summary} {ANSWER_START_MARK}'

    inference_engine = LLMInferenceEngine(llm_path=LLM_FULL_MODEL_PATH,
                                          answer_start_mark=ANSWER_START_MARK,
                                          eos_token=None,
                                          stop_token_list=None)
    inference_engine.load_llm()
    inference_engine.update_tokenizer()
    inference_result = inference_engine.run_inference(full_prompt, epoch=None)
    inference_engine.unload_llm()

    llm_answer = inference_result['llm_answer']
    llm_answer = llm_answer.replace(ANSWER_END_MARK.strip(), '')
    return llm_answer


def simplify_reviewed_files_path(code_review_result_str: str):
    """Simplify Python code file paths to review, especially common prefix."""

    pattern_prefix = '### 파일:'
    pattern = rf'^{pattern_prefix}\s*`([^`]+)`'

    def find_common_prefix(text: str) -> str | None:
        """Find common prefix of all the given paths from the text."""

        file_paths = re.findall(pattern, text, flags=re.MULTILINE)
        file_paths.remove("(코드 전체 경로)")

        if not file_paths:
            return None
        try:
            common_prefix = os.path.commonprefix(file_paths)
            return common_prefix
        except ValueError:
            return None

    common_prefix_of_all_files = find_common_prefix(code_review_result_str)

    if common_prefix_of_all_files:
        return '\n'.join(
            f"{pattern_prefix} " +
            f"`{common_prefix_of_all_files.join(line.split(common_prefix_of_all_files)[1:]).split('`')[0]}`"
            if line.startswith(pattern_prefix) else line
            for line in code_review_result_str.split('\n')
        )
    else:
        return code_review_result_str


def run_entire_process(code_path: str, except_path: str | None = None, verbose: bool = False,
                       llm_final_comment: bool = True) -> dict:

    """Run the entire code review process."""

    code_review_result = run_entire_code_review(code_path, except_path)
    if verbose:
        print_code_review_result(code_review_result)

    top_items_eval_summary = code_review_result[TOP_ITEMS_EVAL_SUMMARY]
    result_dict = {TOP_ITEMS_EVAL_SUMMARY: f"```\n{top_items_eval_summary}\n```",
                   CODE_REVIEW_RESULT_STR: simplify_reviewed_files_path(code_review_result[CODE_REVIEW_RESULT_STR]),
                   MEAN_SCORE: code_review_result[MEAN_SCORE]}

    if llm_final_comment:
        llm_answer = generate_llm_comment(top_items_eval_summary)
        result_dict[LLM_ANSWER] = llm_answer

    return result_dict


if __name__ == '__main__':
    entire_process_result = run_entire_process(code_path=TEST_CASES_DIR)
    entire_process_result_wo_llm = run_entire_process(code_path=TEST_CASES_DIR, llm_final_comment=False)
    entire_process_result_multiple_files = run_entire_process(code_path=TEST_CASES_MULTIPLE_FILES, verbose=True)

    for k, v in entire_process_result.items():
        print(f'\n\n{FINAL_WITH_LLM} [[ {k} ]]')
        print(str(v)[:LENGTH_TO_PRINT])

    for k, v in entire_process_result_wo_llm.items():
        print(f'\n\n{FINAL_WO_LLM} [[ {k} ]]')
        print(str(v)[:LENGTH_TO_PRINT])

    for k, v in entire_process_result_multiple_files.items():
        print(f'\n\n{FINAL_WO_LLM} [[ {k} ]]')
        print(str(v)[:LENGTH_TO_PRINT])
