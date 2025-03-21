
# ORPO 용 데이터셋 생성할 때, 각 LLM output 의 score 평가
# Create Date : 2025.03.21
# Last Update Date : -

# Arguments:
# - eval_preds : Valid data 에 대한 LLM 의 출력 답변 및 실제 목적한 output

# Returns:
# - score (float) : input_data 에 대한 output_data 의 적절성 평가 score

def compute_output_score(eval_preds):
    print(eval_preds)
    raise NotImplementedError


# DataFrame 에 학습 가능한 'text' column 추가
# Create Date : 2025.03.20
# Last Update Date : -

# Arguments:
# - df        (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame
#                                  columns: ['input_data', 'output_data']

# Returns:
# - df 의 'text' column 추가
# - 해당 column 의 형식은 LLM 이 직접 학습 가능한 형태임

def add_text_column_for_llm(df):
    df['text'] = df.apply(lambda x: f"### Question: {x['input_data']}\n ### Answer: {x['output_data']}",
                          axis=1)
