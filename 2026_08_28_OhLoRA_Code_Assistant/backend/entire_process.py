
import os
import sys

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
LLM_FULL_MODEL_PATH = os.path.join(LLM_DIR_PATH, LLM_NAME, 'full_model')


def print_code_review_result(code_review_result: dict):
    for k, v in code_review_result.items():
        if k != 'item_counts':
            print(f'\n\n[ {k} ]')
            print(v)

    print('\n\n[ item_counts ]')
    for rule_id, rule_review_result in code_review_result['item_counts'].items():
        print(rule_id, rule_review_result)


def generate_llm_comment(top_items_eval_summary: str) -> str:
    inference_engine = LLMInferenceEngine(llm_path=LLM_FULL_MODEL_PATH,
                                          answer_start_mark=ANSWER_START_MARK,
                                          eos_token=None,
                                          stop_token_list=None)
    inference_engine.load_llm()
    inference_engine.update_tokenizer()
    inference_result = inference_engine.run_inference(top_items_eval_summary, epoch=None)
    inference_engine.unload_llm()

    llm_answer = inference_result['llm_answer']
    llm_answer = llm_answer.replace(ANSWER_END_MARK.strip(), '')
    return llm_answer


def run_entire_process(code_path: str, verbose: bool = False, llm_final_comment: bool = True) -> dict:
    code_review_result = run_entire_code_review(code_path)
    if verbose:
        print_code_review_result(code_review_result)

    top_items_eval_summary = code_review_result['top_items_eval_summary']
    result_dict = {'top_items_eval_summary': top_items_eval_summary,
                   'code_review_result_str': code_review_result['code_review_result_str'],
                   'mean_score': code_review_result['mean_score']}

    if llm_final_comment:
        llm_answer = generate_llm_comment(top_items_eval_summary)
        result_dict['llm_answer'] = llm_answer

    return result_dict


if __name__ == '__main__':
    entire_process_result = run_entire_process(code_path=TEST_CASES_DIR, verbose=True)

    for k, v in entire_process_result.items():
        print(f'\n\nFINAL [[ {k} ]]')
        print(str(v)[:1000])
