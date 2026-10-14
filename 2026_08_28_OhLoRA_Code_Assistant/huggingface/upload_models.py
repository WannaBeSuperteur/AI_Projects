
import os
import sys

from huggingface_hub import HfApi

PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(PROJECT_DIR_PATH)

from rule_and_model_info import RULES_EMBEDDING, TASK_NAME_TO_INFO


# Most of the code below was generated using ChatGPT-6 Thinking, and later human-reviewed.
# Before upload, there are 30 'model's in https://huggingface.co/daebakgazua (as of '26.10.14 12:00 KST)

api = HfApi()
HF_USERNAME = api.whoami()["name"]

EMBEDDING_REPO_BASE = f"{HF_USERNAME}/260828_OhLoRA_Embedding_"
EMBEDDING_REPOS = [EMBEDDING_REPO_BASE + rule_id for rule_id in RULES_EMBEDDING]
LLM_REPO = f"{HF_USERNAME}/260828_OhLoRA_LLM"
PRIVATE = False

EMBEDDING_MODEL_PATHS = [PROJECT_DIR_PATH / "embedding_model" / "models" / rule_id for rule_id in RULES_EMBEDDING]
LLM_MODEL_PATH = PROJECT_DIR_PATH / "llm_model" / "kanana-2-3b-instruct_fine_tuned" / "full_model"


def upload_casual_llm():
    """Upload Causal LLM. (Transformers, Kanana2-3B Instruct)"""

    assert (LLM_MODEL_PATH / "config.json").is_file()
    assert list(LLM_MODEL_PATH.glob("*.safetensors"))

    api.create_repo(
        repo_id=LLM_REPO,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True
    )
    api.upload_large_folder(
        folder_path=str(LLM_MODEL_PATH),
        repo_id=LLM_REPO,
        repo_type="model"
    )


def upload_embedding_models():
    """Upload 12 embedding models."""

    for path, repo, rule_id in zip(EMBEDDING_MODEL_PATHS, EMBEDDING_REPOS, RULES_EMBEDDING):
        assert path.is_dir(), f"Model directory not found: {path}"
        assert (path / "modules.json").is_file()

        api.create_repo(
            repo_id=repo,
            repo_type="model",
            private=PRIVATE,
            exist_ok=True
        )
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo,
            repo_type="model",
            commit_message=f"Upload {rule_id} fine-tuned SentenceTransformer embedding model"
        )


if __name__ == '__main__':
    upload_casual_llm()
    upload_embedding_models()
