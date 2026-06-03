# LLM to interact with user

## 목차

* [1. 개요](#1-개요)
* [2. LLM 학습 데이터셋 생성 방법](#2-llm-학습-데이터셋-생성-방법)
  * [2-1. Gemini 상세 프롬프트 및 작동 과정](#2-1-gemini-상세-프롬프트-및-작동-과정) 

## 1. 개요

* 하이퍼파라미터 튜닝 대결 결과에 따라, 사용자 상호작용을 위해 LLM이 **평가 대화** 를 출력
* LLM
  * [Kanana-1.5 2.1B Instruct (2025.05)](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)
* [LLM 학습 데이터셋](train_data.csv)
  * 대결 결과에 따른 총 72 rows 규모의 학습 데이터

## 2. LLM 학습 데이터셋 생성 방법

* 처음 14 개의 row는 사람이 직접 입력
* 나머지 58 개의 row는 다음과 같이 **처음 14 개의 row를 참고하여 Gemini로 작성**

### 2-1. Gemini 상세 프롬프트 및 작동 과정

* 최초 (= 1차) 프롬프트

```
F16:F73 을 F2:F15 의 결과값의 말투를 그대로 참고해서, 마찬가지로 아주 자연스럽고 발랄한 말투로 LLM의 출력을 작성해줘 (단, LLM 결과값은 기본적으로 E열의 승패 결과 및 F2:F15에 있는 말투나 단어, 맥락 등을 그대로 사용하되, 학습이 잘 되도록 최대한 다양한 단어, 문체 등을 사용한다. 이모지의 위치도 문장 끝으로 고정하지 않고, 가끔은 중간에도 넣는다.)
```

* 최초 프롬프트에 따른 Gemini 의 추론

```
Generate LLM Outputs
Generate and Populate Text
Analyze the tone and style of existing messages in 시트1!F2:F15 and their relation to 시트1!E2:E15.
Generate 58 new, diverse, and cheerful messages for 시트1!F16:F73 based on the results in 시트1!E16:E73 with dynamic emoji placement.
Write the generated messages into 시트1!F16:F73.
Visually verify the populated text.
```

* 최초 프롬프트에 대한 Gemini 응답

```
LLM 출력 데이터 생성 완료 ✨

E열의 승패 결과와 기존 데이터의 발랄한 말투를 바탕으로, 다양한 어휘와 동적인 이모지 배치를 활용하여 F16:F73 영역에 58개의 새로운 응답 메시지를 생성하고 입력했습니다.
Understanding Your Data
F16:F73 (output_message): E열(input_data)의 각 승패 시나리오에 맞춰 논리적이면서도 기존 F2:F15의 밝고 자연스러운 톤앤매너를 유지하는 응답들이 채워졌습니다.
다양성과 자연스러움: 모델 학습 데이터의 질을 높이기 위해 문체와 단어 선택을 최대한 다양하게 구성했습니다.
이모지 배치: 요청하신 대로 이모지가 문장 끝에만 고정되지 않고, 문장 중간에도 자연스럽게 배치되도록 작성하여 더욱 생동감 있는 문장을 만들었습니다.
✅ Completed Actions

데이터 생성 및 입력
기존 데이터의 말투와 각 행의 승패 맥락을 분석하여 58개의 맞춤형 메시지를 생성했습니다.
생성된 텍스트를 시트의 목표 영역에 성공적으로 반영했습니다.
```

* 추가 (= 2차) 프롬프트

```
데이터 라벨링이 잘못되지 않았으므로 F72를 수정해줘.
```

* 추가 프롬프트에 따른 Gemini 의 추론

```
Update Cell F72
Update Cell F72
Update cell F72 with the new message.
```

* 추가 프롬프트에 대한 Gemini 응답

```
요청하신 대로 F72 셀의 내용을 데이터 라벨링 관련 언급을 제외하고 격려하는 내용으로 수정했습니다.

요약

F72 셀의 텍스트를 "와 이건 좀 심각한데? 🤣 그래도 포기하지 말고 하이퍼파라미터를 다시 잘 설정해 보자! 파이팅!"으로 변경했습니다.
```
