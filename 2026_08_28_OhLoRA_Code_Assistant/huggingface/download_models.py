
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


def download_casual_llm(causal_llm_path: str):
    """Download Causal LLM. (Transformers, Kanana2-3B Instruct)"""

    # TODO implement


def download_embedding_models(embedding_models_root_path: str):
    """Download 12 embedding models."""

    # TODO implement


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path",
                        required=True,
                        help="save path of Causal LLM of Oh-LoRA Code Assistant")
    parser.add_argument("--emb_path",
                        required=True,
                        help="root save path of the Embedding models of Oh-LoRA Code Assistant")

    args = parser.parse_args()

    causal_llm_path = args.llm_path
    embedding_models_root_path = args.emb_path

    download_casual_llm(causal_llm_path)
    download_embedding_models(embedding_models_root_path)

