
import subprocess
import sys
import time
import os

import pandas as pd


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_LOG_PATH = f'{PROJECT_DIR_PATH}/embedding_model/train_log'


TASK_NAMES = [
    "01_unnecessary_prints",
    "01_similar_variables",
    "01_names",
    "01_return_matched_with_func_name",
    "01_func_docstring_single_responsibility",
    "01_func_docstring_docstring_and_name",
    "04_func_args_bindable",
    "04_func_args_dynamic",
    "06_refactor_into_class_case_2_state_vars_if_else",
    "06_similar_function_names",
]


if __name__ == '__main__':
    task_log = {
        'task_name': [],
        'elapsed_time': [],
        'error_msg': []
    }

    for task_name in TASK_NAMES:
        print(f"===== Start: {task_name} =====")
        error_msg = ''
        start_at = time.time()

        try:
            subprocess.run(
                [sys.executable, "train_model.py", "--task", task_name],
                check=True
            )
        except Exception as e:
            error_msg = str(e)

        task_log['task_name'].append(task_name)
        task_log['elapsed_time'].append(round(time.time() - start_at))
        task_log['error_msg'].append(error_msg)
        pd.DataFrame(task_log).to_csv(TRAIN_LOG_PATH)

        print(f"===== Finished: {task_name} =====")
