
import os
import platform
import sys
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from rule_and_model_info import RULES_EMBEDDING


HF_USERNAME = "daebakgazua"

EMBEDDING_REPO_BASE = f"{HF_USERNAME}/260828_OhLoRA_Embedding_"
EMBEDDING_REPOS = [EMBEDDING_REPO_BASE + rule_id for rule_id in RULES_EMBEDDING]
LLM_REPO_ID = f"{HF_USERNAME}/260828_OhLoRA_LLM"


def get_hf_cache_path() -> str:
    if platform.system() == "Windows":
        cache_path = r"C:\ohlora_v7_hf_cache"
    else:
        cache_path = os.path.expanduser("~/ohlora_v7_hf_cache")
    return cache_path


def download_model(hf_repo_id: str, model_save_path: str) -> None:
    temp_path = Path(get_hf_cache_path())
    model_save_path_ = Path(model_save_path).resolve()

    if model_save_path_.exists() and len(os.listdir(model_save_path_)) >= 1:
        print('Model already exists.')
        return

    temp_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="model",
        local_dir=str(temp_path),
    )

    # move model from temp_path to model_save_path_
    os.rename(temp_path, model_save_path_)


def download_casual_llm(causal_llm_save_path: str) -> None:
    """Download Causal LLM. (Transformers, Kanana2-3B Instruct)"""

    os.makedirs(causal_llm_save_path, exist_ok=True)
    download_model(LLM_REPO_ID, causal_llm_save_path)


def download_embedding_models(embedding_models_root_save_path: str) -> None:
    """Download 12 embedding models."""

    for repo_id in EMBEDDING_REPOS:
        rule_id = repo_id.split('_Embedding_')[-1]
        embedding_model_save_path = Path(embedding_models_root_save_path) / rule_id
        os.makedirs(embedding_model_save_path, exist_ok=True)
        download_model(repo_id, embedding_model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path",
                        required=False,
                        default="../llm_model/kanana-2-3b-instruct_fine_tuned/full_model",
                        help="save path of Causal LLM of Oh-LoRA Code Assistant")
    parser.add_argument("--emb_path",
                        required=False,
                        default="../embedding_model/models",
                        help="root save path of the Embedding models of Oh-LoRA Code Assistant")

    args = parser.parse_args()

    causal_llm_save_path = args.llm_path
    embedding_models_root_save_path = args.emb_path

    download_casual_llm(causal_llm_save_path)
    download_embedding_models(embedding_models_root_save_path)

