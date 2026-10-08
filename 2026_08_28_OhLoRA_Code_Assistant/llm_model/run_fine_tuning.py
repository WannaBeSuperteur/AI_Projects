
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


LLM_ORIGINAL_PATHS = ['kanana-2-3b-instruct',
                      'kanana-1.5-2.1b-instruct-2505',
                      'K-intelligence/Midm-2.0-Mini-Instruct',
                      'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B']


class LLMTrainer():
    def __init__(self, original_path: str, save_path: str):
        self.original_path = original_path
        self.save_path = save_path

    def run(self):
        """Train LLM."""

    def save_llm(self):
        """Save LLM into save path."""


def train_and_save_llm(original_path: str, save_path: str):
    llm_trainer = LLMTrainer(original_path, save_path)
    llm_trainer.run()
    llm_trainer.save_llm()


if __name__ == '__main__':
    for original_path in LLM_ORIGINAL_PATHS:
        save_path = os.path.join(PROJECT_DIR_PATH, original_path.split('/')[-1].lower())
        train_and_save_llm(original_path, save_path)
