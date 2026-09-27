import pandas as pd


def remove_newlines_from_columns(
    file_path,
    columns=("code_2", "code")
):
    """CSV 파일의 지정된 컬럼에서 newline 문자를 제거한다."""

    df = pd.read_csv(file_path)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.replace("\r\n", " ")
                           .replace("\n", " ")
                           .replace("\r", " ")
                if isinstance(x, str) else x
            )
    df.to_csv(file_path, index=False)

    print(f"[DONE] {file_path}")


if __name__ == '__main__':

    # 1. docstring_and_name
    remove_newlines_from_columns(file_path="dataset_01_func_docstring_docstring_and_name.csv")

    # 2. single_responsibility
    remove_newlines_from_columns(file_path="dataset_01_func_docstring_single_responsibility.csv")
