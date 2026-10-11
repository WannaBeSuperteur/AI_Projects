
"""
how to run:

cd 2026_08_28_OhLoRA_Code_Assistant/backend
python -m uvicorn run_fastapi:app --host 127.0.0.1 --port 8000
"""
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import redis.asyncio as redis

from entire_process import run_entire_process


app = FastAPI()
redis_client = redis.from_url("redis://localhost:6379?protocol=2", decode_responses=True)

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


def print_as_log(message):
    logger = logging.getLogger("uvicorn.error")
    logger.info(message)


# most of this function code was generated using ChatGPT-6 Thinking, and human-reviewed later.
def get_changed_files(repo_path: Path, base_sha: str, head_sha: str):
    """Get added, modified, or name changed file list, excluding deleted files."""

    merge_base = git("-C", str(repo_path), "merge-base", base_sha, head_sha)
    output = subprocess.check_output([
        "git", "-C", str(repo_path), "diff",
        "--name-only", "--diff-filter=ACMRT", "--find-renames", "-z",
        merge_base, head_sha
    ])

    return [os.fsdecode(name) for name in output.split(b"\x00") if name]


def get_files_to_review(changed_files: list[str], repo_root, except_dir: str):
    files_to_review = []
    seen = set()

    for filename in changed_files:
        if " ||| " in filename:
            raise HTTPException(
                status_code=400,
                detail="Filename contains reserved delimiter"
            )

        relative_path = PurePosixPath(filename)
        if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid changed file path"
            )
        file_path = repo_root.joinpath(
            *relative_path.parts
        )
        resolved_path = file_path.resolve()

        if not resolved_path.is_relative_to(repo_root):
            raise HTTPException(
                status_code=400,
                detail="Changed file escapes repository"
            )

        if resolved_path.is_relative_to(except_dir):
            continue

        if file_path.is_symlink():
            continue

        if not resolved_path.is_file():
            continue

        if resolved_path in seen:
            continue

        seen.add(resolved_path)
        files_to_review.append(str(resolved_path))

    return files_to_review


def get_actual_sha_and_changed_files(repo_path, pr_number: int, commit_hash: str, base_git_ref: str):
    try:
        git("init", str(repo_path))
        git("-C", str(repo_path), "remote", "add", "origin", REPO_URL)
        git("-C", str(repo_path), "fetch", "--depth=1", "origin", f"refs/pull/{pr_number}/head")
        actual_sha = git("-C", str(repo_path), "rev-parse", "FETCH_HEAD")

        if actual_sha.lower() != commit_hash.lower():
            raise HTTPException(
                status_code=409,
                detail="Commit hash mismatch"
            )

        git("-C", str(repo_path), "fetch", "origin", base_git_ref)
        base_sha = git("-C", str(repo_path), "rev-parse", "FETCH_HEAD")
        changed_files = get_changed_files(
            repo_path=repo_path,
            base_sha=base_sha,
            head_sha=actual_sha
        )

        git("-C", str(repo_path), "checkout", "--detach", actual_sha)
        return actual_sha, changed_files

    except HTTPException:
        raise

    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Git operation failed: (exit code {exc.returncode})"
        ) from exc


# most of this function code was generated using ChatGPT-6 Thinking, and human-reviewed later.
@app.get("/review_pr")
def review_pr(commit_hash: str, pr_number: int, base_ref: str, use_llm: bool = True):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit_hash):
        raise HTTPException(
            status_code=400,
            detail="Invalid commit hash"
        )
    if pr_number <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid PR number"
        )

    base_git_ref = f"refs/heads/{base_ref}"
    valid_ref = subprocess.run(
        ["git", "check-ref-format", "--verify", base_git_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if valid_ref.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid base branch"
        )

    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "repo"
        print_as_log(f"Repo Path     : {repo_path.resolve()}")

        actual_sha, changed_files = get_actual_sha_and_changed_files(repo_path, pr_number, commit_hash, base_git_ref)
        print_as_log(f"PR number     : {pr_number}")
        print_as_log(f"Commit Hash   : {commit_hash}")
        print_as_log(f"Base Git Ref  : {base_git_ref}")

        repo_root = repo_path.resolve()
        except_dir = str(Path(repo_root) / "2026_08_28_OhLoRA_Code_Assistant" / "code_reviewer" / "test_cases")
        files_to_review = get_files_to_review(changed_files, repo_root, except_dir)

        print_as_log(f"Except dir    : {except_dir}")
        print_as_log(f"Review files  : {len(files_to_review)}")
        print_as_log(f"              : {files_to_review[:5]}")

        if not files_to_review:
            return {
                "skipped": True,
                "reason": "PR에 리뷰 가능한 변경 파일이 없습니다. (파일 삭제 or 제외 디렉토리 파일만 추가/변경된 경우 등)"
            }

        code_path = " ||| ".join(files_to_review)
        except_path = str(except_dir)

        # final logging
        print_as_log(f"PR #{pr_number}")
        print_as_log(f"Commit        : {actual_sha}")
        print_as_log(f"Changed files : {len(changed_files)}")
        print_as_log(f"                {changed_files[:5]}")
        print_as_log(f"Code path     : {code_path}")
        print_as_log(f"Except path   : {except_path}")

        # AI 코드 리뷰 수행
        result = run_entire_process(code_path=code_path, except_path=except_path, llm_final_comment=use_llm)
        return result
