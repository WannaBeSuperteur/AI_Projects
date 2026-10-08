
import pandas as pd

from utils import LLM_ORIGINAL_PATHS


class LLMInferenceEngine():
    def __init__(self, llm_path: str, top_p: float = 0.95, top_k: int = 50, temperature: float = 0.6):
        self.llm_path = llm_path
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def load_llm(self):
        pass

    def unload_llm(self):
        pass

    def run_inference(self, text_list: list[str]) -> list[str]:
        pass


def load_test_dataset() -> list[str]:
    pass


def save_as_csv(inference_result: list[str], llm_path: str):
    inference_result_dict = {'inference_result': inference_result}
    inference_result_csv = pd.DataFrame(inference_result_dict)
    inference_result_csv.to_csv(f'inference_result_{llm_path}.csv')


if __name__ == '__main__':
    test_dataset = load_test_dataset()

    for llm_original_path in LLM_ORIGINAL_PATHS:
        llm_path = llm_original_path.split('/')[-1].lower()
        llm_inference_engine = LLMInferenceEngine(llm_path=LLM_ORIGINAL_PATHS)

        llm_inference_engine.load_llm()
        inference_result = llm_inference_engine.run_inference(test_dataset)
        llm_inference_engine.unload_llm()

