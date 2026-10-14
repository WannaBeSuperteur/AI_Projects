
import os
import sys

from pathlib import Path
from huggingface_hub import HfApi

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from code_reviewer.rule_info import RULES_EMBEDDING


api = HfApi()
HF_USERNAME = api.whoami()["name"]

EMBEDDING_REPO_BASE = f"{HF_USERNAME}/260828_OhLoRA_Embedding_"
EMBEDDING_REPOS = [EMBEDDING_REPO_BASE + rule_id for rule_id in RULES_EMBEDDING]
LLM_REPO = f"{HF_USERNAME}/260828_OhLoRA_LLM"
PRIVATE = False


def upload_casual_llm():
    """Upload Causal LLM. (Transformers, Kanana2-3B Instruct)"""

    # TODO implement


def upload_embedding_models():
    """Upload 12 embedding models."""

    # TODO implement


if __name__ == '__main__':
    upload_casual_llm()
    upload_embedding_models()
