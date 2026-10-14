
import os
import sys

from pathlib import Path
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
LLM_REPO_ID = f"{HF_USERNAME}/260828_OhLoRA_LLM"
PRIVATE = False

PROJECT_DIR_PATH_ = Path(PROJECT_DIR_PATH)
EMBEDDING_MODEL_PATHS = [PROJECT_DIR_PATH_ / "embedding_model" / "models" / rule_id for rule_id in RULES_EMBEDDING]
LLM_MODEL_PATH = (PROJECT_DIR_PATH_ / "llm_model" / "kanana-2-3b-instruct_fine_tuned" / "full_model")
README_TEMPLATE_PATH = (PROJECT_DIR_PATH_ / "huggingface" / "readme_templates")


def upload_casual_llm():
    """Upload Causal LLM. (Transformers, Kanana2-3B Instruct)"""

    assert (LLM_MODEL_PATH / "config.json").is_file()
    assert list(LLM_MODEL_PATH.glob("*.safetensors"))

    api.create_repo(
        repo_id=LLM_REPO_ID,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True
    )
    api.upload_large_folder(
        folder_path=str(LLM_MODEL_PATH),
        repo_id=LLM_REPO_ID,
        repo_type="model"
    )

    local_readme_teamplate_path = Path(README_TEMPLATE_PATH) / "llm_template.txt"
    local_readme = local_readme_teamplate_path.read_text(encoding="utf-8")

    api.upload_file(
        path_or_fileobj=local_readme.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=LLM_REPO_ID,
        repo_type="model",
        commit_message="Update model card",
    )


def upload_embedding_models():
    """Upload 12 embedding models."""

    for local_path, hf_repo_id, rule_id in zip(EMBEDDING_MODEL_PATHS, EMBEDDING_REPOS, RULES_EMBEDDING):
        assert local_path.is_dir(), f"Model directory not found: {local_path}"
        assert ((local_path / "modules.json").is_file() or (local_path / "model.safetensors").is_file())

        api.create_repo(
            repo_id=hf_repo_id,
            repo_type="model",
            private=PRIVATE,
            exist_ok=True
        )
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=f"Upload {rule_id} fine-tuned SentenceTransformer embedding model"
        )

        local_readme_teamplate_path = README_TEMPLATE_PATH / "embedding_model_template.txt"
        local_readme = local_readme_teamplate_path.read_text(encoding="utf-8")

        local_readme = local_readme.replace('{base_model}', TASK_NAME_TO_INFO[rule_id]['model_path'])
        local_readme = local_readme.replace('{rule_id}', rule_id)
        local_readme = local_readme.replace('{what_to_predict}', TASK_NAME_TO_INFO[rule_id]['task_type'])

        api.upload_file(
            path_or_fileobj=local_readme.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message="Update model card",
        )


if __name__ == '__main__':
    upload_casual_llm()
    upload_embedding_models()
