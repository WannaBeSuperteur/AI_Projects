
"""
how to run:

cd 2026_08_28_OhLoRA_Code_Assistant/backend
python -m uvicorn run_fastapi:app --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI
from entire_process import run_entire_process


app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.get("/review_code")
def run_code_review(code_path: str, use_llm: bool):
    entire_process_result = run_entire_process(code_path=code_path, llm_final_comment=use_llm)
    return entire_process_result

