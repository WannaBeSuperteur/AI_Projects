from typing import Any

import torch
import os
import gc
import time

import pandas as pd
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer
from utils import add_inference_log


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# stop when "LAST N TOKENS MATCHES stop_token_ids" - class code by ChatGPT-4o
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = list(stop_token_ids.detach().cpu().numpy())
        self.current_ids = []

    def __call__(self, input_ids, scores, **kwargs):
        self.current_ids = input_ids[0].tolist()

        if len(self.current_ids) >= len(self.stop_token_ids):
            if self.current_ids[-len(self.stop_token_ids):] == self.stop_token_ids:
                return True  # stop generation

        return False


class LLMInferenceEngine:
    def __init__(self, llm_path: str | None, answer_start_mark: str, eos_token: str, stop_token_list: list[int],
                 top_p: float = 0.95, top_k: int = 50, temperature: float = 0.6,
                 inference_log_dict: dict | None = None):

        self.llm_path = llm_path
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

        self.answer_start_mark = answer_start_mark
        self.eos_token = eos_token
        self.stop_token_list = stop_token_list

        self.fine_tuned_llm = None
        self.tokenizer = None

        if inference_log_dict is not None:
            self.inference_log_dict = inference_log_dict
        else:
            self.inference_log_dict = {'epoch': [],
                                       'elapsed_time (s)': [],
                                       'prompt': [],
                                       'llm_answer': [],
                                       'trial_cnt': [],
                                       'output_tkn_cnt': [],
                                       'torch_memory_kb': []}

    def load_fine_tuned_llm_directly(self, fine_tuned_llm, tokenizer):
        self.fine_tuned_llm = fine_tuned_llm
        self.tokenizer = tokenizer

    def load_llm(self):
        if self.fine_tuned_llm is not None and self.tokenizer is not None:
            print("LLM already loaded (or directly loaded)")
            return

        self.fine_tuned_llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)

    def unload_llm(self):
        if self.fine_tuned_llm is None:
            print("LLM not loaded")
            return

        self.fine_tuned_llm = None
        self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

    def _run_inference(self, prompt: str, max_new_tokens: int = 80, max_trials: int = 5,
                       additional_answer_test_func: callable = None) -> dict:

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.fine_tuned_llm.generation_config.pad_token_id = self.tokenizer.pad_token_id

        final_input_prompt = prompt + self.answer_start_mark
        inputs = self.tokenizer(final_input_prompt, return_tensors='pt').to(self.fine_tuned_llm.device)
        input_token_cnt = inputs['input_ids'].shape[1]

        llm_answer = ''
        trial_cnt = 0
        new_token_cnt, total_token_cnt = None, None

        # for stopping criteria
        stop_token_ids = torch.tensor(self.stop_token_list).to(self.fine_tuned_llm.device)
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        while trial_cnt < max_trials:
            outputs = self.fine_tuned_llm.generate(**inputs,
                                                   max_new_tokens=max_new_tokens,
                                                   do_sample=True,
                                                   temperature=self.temperature,
                                                   stopping_criteria=stopping_criteria)
            total_token_cnt = len(outputs[0])
            new_token_cnt = total_token_cnt - input_token_cnt

            llm_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            llm_answer = llm_answer[len(final_input_prompt):]
            trial_cnt += 1

            # check LLM answer and return or retry
            is_non_empty = llm_answer.replace('\n', '').replace(self.eos_token, '').replace(' ', '') != ''
            is_acceptable = (is_non_empty and
                             (additional_answer_test_func is None or additional_answer_test_func(llm_answer)))

            if is_acceptable:
                break

        # remove new-lines
        llm_answer = llm_answer.replace('\n', '')

        return {'llm_answer': llm_answer,
                'trial_cnt': trial_cnt,
                'total_token_cnt': total_token_cnt,
                'input_token_cnt': input_token_cnt,
                'new_token_cnt': new_token_cnt}

    def run_inference(self, prompt: str, epoch: Any, load_and_unload_llm: bool = True):
        start_at = time.time()

        if load_and_unload_llm:
            self.load_llm()
        inference_result = self._run_inference(prompt)
        if load_and_unload_llm:
            self.unload_llm()

        llm_answer, trial_cnt, total_token_cnt, input_token_cnt, new_token_cnt = (
            inference_result['llm_answer'], inference_result['trial_cnt'],
            inference_result['total_token_cnt'], inference_result['input_token_cnt'], inference_result['new_token_cnt'])

        llm_answer = llm_answer[:-len(self.eos_token) + 1]
        elapsed_time = time.time() - start_at

        print(f'input prompt : {prompt}')
        print(f'llm answer (trials: {trial_cnt}, new tokens: {new_token_cnt}) : {llm_answer}\n')

        inference_result = {'epoch': epoch,
                            'elapsed_time': elapsed_time,
                            'prompt': prompt,
                            'llm_answer': llm_answer,
                            'trial_cnt': trial_cnt,
                            'total_token_cnt': total_token_cnt,
                            'input_token_cnt': input_token_cnt,
                            'new_token_cnt': new_token_cnt,
                            'torch_memory_kb': torch.cuda.memory_allocated() // 1024}

        add_inference_log(inference_result, self.inference_log_dict)


def save_as_csv(inference_result: list[str], llm_path: str):
    inference_result_dict = {'inference_result': inference_result}
    inference_result_csv = pd.DataFrame(inference_result_dict)
    inference_result_csv.to_csv(f'inference_result_{llm_path}.csv')

