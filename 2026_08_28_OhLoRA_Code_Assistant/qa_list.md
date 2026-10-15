# Oh-LoRA Code Assistant QA List

## 목차

* [1. 기본 사항](#1-기본-사항)
* [2. QA 리스트](#2-QA-리스트)
  * [2-1. 테스트 케이스 목록](#2-1-테스트-케이스-목록) 
* [3. 버그 수정 로그](#3-버그-수정-로그)
  * [3-1. 모델 다운로드 시 경로 찾기 불가](#3-1-모델-다운로드-시-경로-찾기-불가) 

## 1. 기본 사항

* 본 문서는 **Oh-LoRA 👱‍♀️ (오로라) Code Assistant** 에 대한 체계적인 검증을 위한 QA List 이다.
* 1차 검사 통과 시, **1차 검사 결과** 를 최종 결과로 하고 **최종 검사 결과** 란은 비운다.
* 검사 결과는 이모지 (✅ or ❌) 로 나타낸다.

## 2. QA 리스트

| 번호 | 검사 항목                                                                                                                                                  | 1차 검사 결과 | 버그 수정                               | 최종 검사 결과 |
|----|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-------------------------------------|----------|
| 1  | 모델 다운로드 정상 동작 (사용자 PC 기준)                                                                                                                              | ❌        | [버그 수정 로그](#3-1-모델-다운로드-시-경로-찾기-불가) | ✅        |
| 2  | API 실행 시 정상 동작 (서버 PC 기준)                                                                                                                              | ✅        |                                     |          |
| 3  | [사용자 프로그램](user_program/main.py) 테스트 케이스 실행 결과 일치                                                                                                      | ✅        |                                     |          |
| 4  | [code_reviewer.py](code_reviewer/code_reviewer.py) 실행 시 임베딩 로그가 [기존 모델 로그](code_reviewer/bugfix_logs/ai_code_check_log_before_model_download.csv) 와 동일 |          |                                     |          |

### 2-1. 테스트 케이스 목록

* 전체 테스트 케이스는 [사용자 프로그램 코드 파일](user_program/main.py) 의 docstring 참고.

| 번호  | 테스트 목적                       | 테스트 케이스 (Windows shell 명령어)                                                                                                                                                      | 1차 검사 결과 | 버그 수정 | 최종 검사 결과 |
|-----|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-------|----------|
| 3-1 | 기본 형태 (코드 경로)                | `python main.py --code_path ../code_reviewer/test_cases --verbose True`                                                                                                          | ✅        |       |          |
| 3-2 | `except_path` (제외 경로) 포함     | `python main.py --code_path ../code_reviewer/test_cases --except_path ../code_reviewer/test_cases/path_test --verbose True`                                                      | ✅        |       |          |
| 3-3 | 단일 파일 테스트                    | `python main.py --code_path ../code_reviewer/test_cases/tc02.py --verbose True`                                                                                                  | ✅        |       |          |
| 3-4 | 다중 파일 테스트                    | `python main.py --code_path "../code_reviewer/test_cases/tc02.py \|\|\| ../code_reviewer/test_cases/tc03.py" --verbose True`                                                     | ✅        |       |          |
| 3-5 | 단일 파일 테스트 (`except_path` 포함) | `python main.py --code_path ../code_reviewer/test_cases/tc02.py --except_path ../code_reviewer/test_cases/path_test --verbose True`                                              | ✅        |       |          |
| 3-6 | 다중 파일 테스트 (`except_path` 포함) | `python main.py --code_path "../code_reviewer/test_cases/tc02.py \|\|\| ../code_reviewer/test_cases/tc03.py" --except_path ../code_reviewer/test_cases/path_test --verbose True` | ✅        |       |          |

## 3. 버그 수정 로그

### 3-1. 모델 다운로드 시 경로 찾기 불가

* 오류 상황

```
PS C:\Users\20151\Documents\AI_Projects_Test\AI_Projects\2026_08_28_OhLoRA_Code_Assistant\huggingface> python download_models.py
Fetching 15 files:   0%|                                                                                                                                                                                    | 0/15 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "download_models.py", line 63, in <module>
    download_casual_llm(causal_llm_save_path)
  File "download_models.py", line 34, in download_casual_llm
    download_llm(LLM_REPO_ID, causal_llm_save_path)
  File "download_models.py", line 23, in download_llm
    snapshot_download(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\huggingface_hub\_snapshot_download.py", line 296, in snapshot_download
    thread_map(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\tqdm\contrib\concurrent.py", line 69, in thread_map
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\tqdm\contrib\concurrent.py", line 51, in _executor_map
    return list(tqdm_class(ex.map(fn, *iterables, chunksize=chunksize), **kwargs))
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\tqdm\std.py", line 1181, in __iter__
    for obj in iterable:
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\concurrent\futures\_base.py", line 611, in result_iterator
    yield fs.pop().result()
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\concurrent\futures\_base.py", line 439, in result
    return self.__get_result()
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\concurrent\futures\_base.py", line 388, in __get_result
    raise self._exception
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\concurrent\futures\thread.py", line 57, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\huggingface_hub\_snapshot_download.py", line 270, in _inner_hf_hub_download
    return hf_hub_download(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\huggingface_hub\file_download.py", line 842, in hf_hub_download
    return _hf_hub_download_to_local_dir(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\huggingface_hub\file_download.py", line 1138, in _hf_hub_download_to_local_dir
    _download_to_tmp_and_move(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\huggingface_hub\file_download.py", line 1535, in _download_to_tmp_and_move
    with incomplete_path.open("ab") as f:
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\pathlib.py", line 1210, in open
    return io.open(self, mode, buffering, encoding, errors, newline,
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\pathlib.py", line 1066, in _opener
    return self._accessor.open(self, flags, mode)
FileNotFoundError: [Errno 2] No such file or directory: '..\\llm_model\\kanana-2-3b-instruct_fine_tuned\\full_model\\.cache\\huggingface\\download\\wPaCkH-WbT7GsmxMKKrNZTV4nSM=.a6344aac8c09253b3b630fb776ae94478aa0275b.incomplete'
```

* 오류 원인
  * Windows 에서 HuggingFace Cache 저장 경로의 **path 길이가 260자 초과 (약 267자)** 로 오류 발생

* 해결 방법
  * Cache 위치를 임시로 수정 (OS 별 적용) → **❌ 해결 안됨**
  * 모델 Cache 저장 경로를 '길이가 짧은' 임시 경로로 지정, 이후 모델 이동 → **✅ 해결 완료**

```python
# 모델 Cache 저장 경로를 '길이가 짧은' 임시 경로로 지정, 이후 모델 이동

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
    model_save_path_.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="model",
        local_dir=str(temp_path),
    )
    
    # move model from temp_path to model_save_path_
    os.rename(temp_path, model_save_path_)
```

