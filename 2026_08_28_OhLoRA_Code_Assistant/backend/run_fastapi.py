
"""
how to run:

cd 2026_08_28_OhLoRA_Code_Assistant/backend
python -m uvicorn run_fastapi:app --host 127.0.0.1 --port 8000
"""
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import redis.asyncio as redis

from entire_process import run_entire_process


app = FastAPI()
redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)

REPO_URL = "https://github.com/WannaBeSuperteur/AI_Projects.git"

LIMIT_WINDOW = 60
ERROR_LIMIT = 2
BLOCK_DURATION = 1800


# most of this function code was generated using Google Search AI, and human-reviewed later.
@app.middleware("http")
async def ban_scanner_with_redis_middleware(request: Request, call_next):
    client_ip = request.client.host

    block_key = f"block:{client_ip}"
    count_key = f"404_count:{client_ip}"

    is_blocked = await redis_client.get(block_key)
    if is_blocked:
        return JSONResponse(
            status_code=403,
            content={"detail": "Access denied due to suspicious activity."}
        )

    response = await call_next(request)
    if response.status_code == 404:
        current_count = await redis_client.incr(count_key)
        if current_count == 1:
            await redis_client.expire(count_key, LIMIT_WINDOW)

        if current_count >= ERROR_LIMIT:
            await redis_client.set(block_key, "1", ex=BLOCK_DURATION)
            await redis_client.delete(count_key)
            print(f"🚨 [Redis ALERT] IP blocked: {client_ip} (for {BLOCK_DURATION // 60} minutes)")

    return response


# 앱 종료 시 Redis 연결 안전 종료
@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.get("/review_code")
def run_code_review(code_path: str, use_llm: bool):
    entire_process_result = run_entire_process(code_path=code_path, llm_final_comment=use_llm)
    return entire_process_result


@app.get("/review_code_with_except")
def run_code_review_with_except_code(code_path: str, except_path: str, use_llm: bool):
    entire_process_result = run_entire_process(code_path=code_path, except_path=except_path, llm_final_comment=use_llm)
    return entire_process_result


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


# most of this function code was generated using ChatGPT-6 Thinking, and human-reviewed later.
@app.get("/review_pr")
def review_pr(commit_hash: str, pr_number: int, use_llm: bool = True):
    with tempfile.TemporaryDirectory() as tmp:
        code_path = Path(tmp) / "repo"

        git("init", str(code_path))
        git("-C", str(code_path), "remote", "add", "origin", REPO_URL)
        git("-C", str(code_path), "fetch", "--depth=1", "origin", f"refs/pull/{pr_number}/head")
        actual_sha = git("-C", str(code_path), "rev-parse", "FETCH_HEAD")

        if actual_sha.lower() != commit_hash.lower():
            raise HTTPException(
                status_code=409,
                detail="Commit hash mismatch"
            )

        git("-C", str(code_path), "checkout", "--detach", actual_sha)

        code_path = str(code_path)
        except_path = str(Path(code_path) / "code_reviewer" / "test_cases")

        # AI 코드 리뷰 수행
        result = run_entire_process(code_path=code_path, except_path=except_path, llm_final_comment=use_llm)
        return result
