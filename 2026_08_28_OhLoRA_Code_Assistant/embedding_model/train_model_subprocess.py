
import subprocess
import sys
import time
import os

import pandas as pd
import torch


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRAIN_SCRIPT_PATH = f'{PROJECT_DIR_PATH}/embedding_model/train_model.py'
TRAIN_LOG_PATH = f'{PROJECT_DIR_PATH}/embedding_model/train_log'

MAX_TRIAL_COUNT = 5


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
    os.makedirs(TRAIN_LOG_PATH, exist_ok=True)
    task_log_path = os.path.join(TRAIN_LOG_PATH, "error_log.csv")

    task_log = {
        'task_name': [],
        'trial_no': [],
        'elapsed_time': [],
        'error_msg': [],
        'cuda_memory': []
    }

    for task_name in TASK_NAMES:
        print(f"===== Start: {task_name} =====")
        error_msg = ''
        current_trial = 0

        while current_trial < MAX_TRIAL_COUNT:
            is_successful = False
            start_at = time.time()

            try:
                process = subprocess.run(
                    [sys.executable, "-u", TRAIN_SCRIPT_PATH, "--task", task_name],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )
                is_successful = True

            except subprocess.CalledProcessError as error:
                lines = (error.stderr or "").strip().splitlines()
                error_msg = lines[-1] if lines else str(error)

            task_log['task_name'].append(task_name)
            task_log['trial_no'].append(current_trial + 1)
            task_log['elapsed_time'].append(round(time.time() - start_at, 3))
            task_log['error_msg'].append(error_msg)
            task_log['cuda_memory'].append(torch.cuda.memory_allocated())
            pd.DataFrame(task_log).to_csv(task_log_path)

            if is_successful:
                break

            current_trial += 1

        print(f"===== Finished: {task_name} =====")
