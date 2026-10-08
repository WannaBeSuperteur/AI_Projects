
import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


LLM_ORIGINAL_PATHS = ['kakaocorp/kanana-2-3b-instruct',
                      'kakaocorp/kanana-1.5-2.1b-instruct-2505',
                      'K-intelligence/Midm-2.0-Mini-Instruct',
                      'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B']

TARGET_MODULES_DICT = {
    'default': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

DATASET_PATH = f'{PROJECT_DIR_PATH}/llm/train_data.csv'
