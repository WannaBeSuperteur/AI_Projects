
def get_instruction():
    return '당신은 AI 여성 챗봇입니다. 사용자의 대화에 답하세요.'


# Modified Implementation from https://github.com/quantumaikr/KoreanLM/blob/main/finetune-lora.py

def koreanlm_tokenize(prompt, tokenizer, return_tensors):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=96,
        padding=False,
        return_tensors=return_tensors,
    )

    result["labels"] = result["input_ids"].copy()
    return result
