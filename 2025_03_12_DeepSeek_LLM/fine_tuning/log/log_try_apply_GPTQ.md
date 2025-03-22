
## 목차

* [1. 개요](#1-개요)
* [2. Fine-Tuning 된 Non-GPTQ 모델에 GPTQ 적용 (실패)](#2-fine-tuning-된-non-gptq-모델에-gptq-적용-실패)
* [3. 처음부터 GPTQ 적용하여 모델 학습 (실패)](#3-처음부터-gptq-적용하여-모델-학습-실패)
  * [3-1. GPTQ 적용만 했을 때](#3-1-gptq-적용만-했을-때)
  * [3-2. ```python -m bitsandbytes``` 실행 결과](#3-2-python--m-bitsandbytes-실행-결과)
  * [3-3. bitsandbytes 0.41.0 -> 0.45.3 업그레이드 이후](#3-3-bitsandbytes-0410---0453-업그레이드-이후)
  * [3-4. ```training_args``` (학습 argument) 수정 이후](#3-4-trainingargs-학습-argument-수정-이후)
  * [3-5. device_map 추가](#3-5-devicemap-추가)
  * [3-6. hf_device_map 직접 설정 시도 (불가능)](#3-6-hfdevicemap-직접-설정-시도-불가능)
* [4. DeepSeek 모델의 GPTQ 버전 이용 (추론속도 향상 안됨)](#4-deepseek-모델의-gptq-버전-이용-추론속도-향상-안됨)
  * [4-1. GPTQ 적용만 했을 때](#4-1-gptq-적용만-했을-때)
  * [4-2. optimum 라이브러리 최신 버전 설치 후](#4-2-optimum-라이브러리-최신-버전-설치-후)
  * [4-3. QLoRA 처리](#4-3-qlora-처리)
  * [4-4. ```device_map='cuda'``` 추가](#4-4-devicemapcuda-추가)

## 1. 개요

* [Fine-Tuning 된 모델의 추론 속도 저하 문제](../../README.md#5-6-fine-tuning-된-모델-추론-속도-저하-해결-보류) 문제를 [GPTQ](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/LLM%20Basics/LLM_%EA%B8%B0%EC%B4%88_Quantization.md#2-4-gptq-post-training-quantization-for-gpt-models) Quantization 을 적용하여 해결하려고 했음
* 다음과 같이 3가지 방법으로 시도했으나, 결과적으로 **추론 속도 향상에 실패** 함

| 시도한 해결 방법                          | 결과                                |
|------------------------------------|-----------------------------------|
| Fine-Tuning 된 Non-GPTQ 모델에 GPTQ 적용 | 실패 (오류 발생)                        |
| 처음부터 GPTQ 적용하여 모델 학습               | 실패 (오류 발생)                        |
| DeepSeek 모델의 GPTQ 버전 이용            | 오류는 없음 (그러나, 학습 및 추론 속도 향상 체감 불가) |

**시스템 환경**

* OS: **Windows 10**
* Python: **Python 3.8.1rc1**
* GPU: **2 x Quadro 6000 (12GB each)**
* 실행 환경: **PyCharm 2024.1 (Community Edition)**

## 2. Fine-Tuning 된 Non-GPTQ 모델에 GPTQ 적용 (실패)

```
loading LLM failed : Unrecognized model in sft_model. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: albert, align, altclip, audio-spectrogram-transformer, autoformer, bar
k, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, blenderbot, blenderbot-small, blip, blip-2, bloom, bridgetower, bros, camembert, canine, chameleon, chinese_clip, chinese_clip_vision_model, clap, cli
p, clip_text_model, clip_vision_model, clipseg, clvp, code_llama, codegen, cohere, conditional_detr, convbert, convnext, convnextv2, cpmant, ctrl, cvt, dac, data2vec-audio, data2vec-text, data2vec-vision, dbrx, deberta, deberta-
v2, decision_transformer, deformable_detr, deit, depth_anything, deta, detr, dinat, dinov2, distilbert, donut-swin, dpr, dpt, efficientformer, efficientnet, electra, encodec, encoder-decoder, ernie, ernie_m, esm, falcon, falcon_
mamba, fastspeech2_conformer, flaubert, flava, fnet, focalnet, fsmt, funnel, fuyu, gemma, gemma2, git, glm, glpn, gpt-sw3, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gpt_neox_japanese, gptj, gptsan-japanese, granite, granitemoe, grap
hormer, grounding-dino, groupvit, hiera, hubert, ibert, idefics, idefics2, idefics3, imagegpt, informer, instructblip, instructblipvideo, jamba, jetmoe, jukebox, kosmos-2, layoutlm, layoutlmv2, layoutlmv3, led, levit, lilt, llam
a, llava, llava_next, llava_next_video, llava_onevision, longformer, longt5, luke, lxmert, m2m_100, mamba, mamba2, marian, markuplm, mask2former, maskformer, maskformer-swin, mbart, mctct, mega, megatron-bert, mgp-str, mimi, mis
tral, mixtral, mllama, mobilebert, mobilenet_v1, mobilenet_v2, mobilevit, mobilevitv2, moshi, mpnet, mpt, mra, mt5, musicgen, musicgen_melody, mvp, nat, nemotron, nezha, nllb-moe, nougat, nystromformer, olmo, olmoe, omdet-turbo,
 oneformer, open-llama, openai-gpt, opt, owlv2, owlvit, paligemma, patchtsmixer, patchtst, pegasus, pegasus_x, perceiver, persimmon, phi, phi3, phimoe, pix2struct, pixtral, plbart, poolformer, pop2piano, prophetnet, pvt, pvt_v2,
 qdqbert, qwen2, qwen2_audio, qwen2_audio_encoder, qwen2_moe, qwen2_vl, rag, realm, recurrent_gemma, reformer, regnet, rembert, resnet, retribert, roberta, roberta-prelayernorm, roc_bert, roformer, rt_detr, rt_detr_resnet, rwkv,
 sam, seamless_m4t, seamless_m4t_v2, segformer, seggpt, sew, sew-d, siglip, siglip_vision_model, speech-encoder-decoder, speech_to_text, speech_to_text_2, speecht5, splinter, squeezebert, stablelm, starcoder2, superpoint, swiftf
ormer, swin, swin2sr, swinv2, switch_transformers, t5, table-transformer, tapas, time_series_transformer, timesformer, timm_backbone, trajectory_transformer, transfo-xl, trocr, tvlt, tvp, udop, umt5, unispeech, unispeech-sat, un
ivnet, upernet, van, video_llava, videomae, vilt, vipllava, vision-encoder-decoder, vision-text-dual-encoder, visual_bert, vit, vit_hybrid, vit_mae, vit_msn, vitdet, vitmatte, vits, vivit, wav2vec2, wav2vec2-bert, wav2vec2-conformer, wavlm, whisper, xclip, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xmod, yolos, yoso, zamba, zoedepth
```

## 3. 처음부터 GPTQ 적용하여 모델 학습 (실패)

**시도 결과 요약**

* ```hf_device_map``` 설정 실패로 인해 해당 방법은 **적용 실패**
* 시도한 방법들
  * bitsandbytes 라이브러리 버전 0.45.3 으로 업그레이드 **(이 방법으로 해결 실패)**
  * ```training_args``` 학습 argument 수정 **(이 방법으로 해결 실패)**
  * ```device_map``` 추가 **(이 방법으로 해결 실패)**
  * ```hf_device_map``` 이 없다는 오류를 해결하기 위해 직접 해당 변수 값 설정 시도 **(불가능)**

### 3-1. GPTQ 적용만 했을 때

**CUDA Setup failed despite GPU being available** 오류 발생

```
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\bitsandbytes\cextension.py", line 20, in <module>
    raise RuntimeError('''
RuntimeError:
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```

### 3-2. ```python -m bitsandbytes``` 실행 결과

* 아래와 같은 오류 발생
* CUDA version 을 ```nvcc -V``` 로 표시되는 것과 ```nvidia-smi``` 로 표시되는 것을 12.4 로 일치시켰는데도 동일 오류 발생
* 재부팅 이후에도 동일 오류 발생

```
================================================ERROR=====================================
CUDA SETUP: CUDA detection failed! Possible reasons:
1. You need to manually override the PyTorch CUDA version. Please see: "https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md
2. CUDA driver not installed
3. CUDA not installed
4. You have multiple conflicting CUDA libraries
5. Required library not pre-compiled for this bitsandbytes release!
CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.
CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.
================================================================================
```

### 3-3. bitsandbytes 0.41.0 -> 0.45.3 업그레이드 이후

* 참고: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1568
* 아래와 같은 오류 발생

```
ValueError: No columns in the dataset match the model's forward method signature. The following columns have been ignored: [attention_mask, input_ids]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.
```

### 3-4. ```training_args``` (학습 argument) 수정 이후

* training argument 를 다음과 같이 수정
  * ```remove_unused_columns=False``` 추가 

```python
    training_args = SFTConfig(
        learning_rate=0.0002,  # lower learning rate is recommended for fine tuning
        num_train_epochs=4,
        logging_steps=1,  # logging frequency
        gradient_checkpointing=True,
        output_dir=output_dir,
        save_total_limit=3,  # max checkpoint count to save
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1,  # batch size per device during validation
        remove_unused_columns=False  # to prevent GPTQ error
    )
```

* 결과: 아래와 같은 오류 발생

```
Traceback (most recent call last):
  File "fine_tuning/sft_fine_tuning.py", line 222, in <module>
    llm, tokenizer = run_fine_tuning(df_train, df_valid)
  File "fine_tuning/sft_fine_tuning.py", line 103, in run_fine_tuning
    trainer.train()
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\trl\trainer\sft_trainer.py", line 434, in train
    output = super().train(*args, **kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\trainer.py", line 2123, in train
    return inner_training_loop(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\trainer.py", line 2275, in _inner_training_loop
    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\accelerate\accelerator.py", line 1296, in prepare
    and self.verify_device_map(obj)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\accelerate\accelerator.py", line 3573, in verify_device_map
    if hasattr(m, "hf_device_map") and len(m.hf_device_map) > 1:
TypeError: object of type 'NoneType' has no len()
```

### 3-5. device_map 추가

* ```device_map='cuda'``` 추가

```python
    original_llm = AutoGPTQForCausalLM.from_pretrained(model_path,
                                                       quantize_config=quantize_config,
                                                       torch_dtype=torch.float16,
                                                       device_map='cuda')
```

* 결과 : **3-4 와 동일한 오류 발생**

### 3-6. hf_device_map 직접 설정 시도 (불가능)

* ```hf_device_map=``` 변수의 값을 인수로 추가하여 직접 설정 시도

```python
    original_llm = AutoGPTQForCausalLM.from_pretrained(model_path,
                                                       quantize_config=quantize_config,
                                                       torch_dtype=torch.float16,
                                                       hf_device_map={'': 0})
```

* 결과 : **해당 변수의 값은 지정 불가능**

## 4. DeepSeek 모델의 GPTQ 버전 이용 (추론속도 향상 안됨)

**개요**

* [기존 모델의 GPTQ 적용된 버전 (TheBloke/deepseek-coder-1.3b-instruct-GPTQ)](https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GPTQ) 적용 시도
* 결과적으로, 시도 자체는 성공했고 GPU 메모리 감소는 확인했으나, **학습 및 추론 속도 향상이 체감되지 않음**
* 메모리 문제 해결보다는 **학습 및 추론 속도 향상이 문제 해결의 목표** 라는 점에서 **사실상 해결 실패**

**시도 결과 요약**

* 적용 자체 (런타임 오류 없이 실행) 는 **성공**
* 그러나, GPU 메모리 사용량 감소는 체감되지만, **학습/추론 속도 개선은 체감 안됨**
  * 기존 모델의 **약 1/4 수준**, 즉 1.7 ~ 2.0 GB 정도로 감소
* 기존 모델 대비 파라미터 개수 감소로, 실제 유저 사용 시 성능 저하 우려
  * 기존 모델 약 1.3B params 에서 약 130M params 로, **1/10 수준으로 감소**
* 추론 속도가 **기존 모델보다도 오히려 느린** 것으로 의심

<details><summary>DeepSeek 모델의 GPTQ 버전을 적용한 SFT Fine-tuning 학습/추론 코드 (fine_tuning/sft_fine_tuning.py)</summary>

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from datasets import DatasetDict, Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.model_selection import train_test_split
from common import compute_output_score, add_text_column_for_llm
from draw_diagram.draw_diagram import generate_diagram_from_lines
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import pandas as pd
import numpy as np
import time


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# to prevent "RuntimeError: chunk expects at least a 1-dimensional tensor"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = '1'


# SFT 실시
# Create Date : 2025.03.21
# Last Update Date : 2025.03.22
# - Auto-GPTQ 적용

# Arguments:
# - df_train (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Train Data
#                                 columns: ['input_data', 'output_data', 'dest_shape_info']
# - df_valid (Pandas DataFrame) : 학습 데이터셋 csv 파일로부터 얻은 DataFrame 중 Valid Data
#                                 columns: ['input_data', 'output_data', 'dest_shape_info']

# Returns:
# - lora_llm  (LLM)       : SFT 로 Fine-tuning 된 LLM
# - tokenizer (tokenizer) : 해당 LLM 에 대한 tokenizer

def run_fine_tuning(df_train, df_valid):

    print(f'Train DataFrame:\n{df_train}')
    print(f'Valid DataFrame:\n{df_valid}')

    model_path = "TheBloke/deepseek-coder-1.3b-instruct-GPTQ"
    output_dir = "sft_model"

    original_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                        device_map='cuda',
                                                        torch_dtype=torch.float16)
    original_llm.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_path, eos_token='<eos>')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    lora_config = LoraConfig(
        r=16,  # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,  # Dropout for LoRA
        init_lora_weights="gaussian",  # LoRA weight initialization
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
    )
    quantized_llm = prepare_model_for_kbit_training(original_llm)
    lora_llm = get_peft_model(quantized_llm, lora_config)
    lora_llm.print_trainable_parameters()

    training_args = SFTConfig(
        learning_rate=0.0002,  # lower learning rate is recommended for fine tuning
        num_train_epochs=4,
        logging_steps=1,  # logging frequency
        gradient_checkpointing=True,
        output_dir=output_dir,
        save_total_limit=3,  # max checkpoint count to save
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1  # batch size per device during validation
    )

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(df_train)
    dataset['valid'] = Dataset.from_pandas(df_valid)

    response_template = '### Answer:'
    response_template = tokenizer.encode(f"\n{response_template}", add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        lora_llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        dataset_text_field='text',
        tokenizer=tokenizer,
        max_seq_length=1536,  # 한번에 입력 가능한 최대 token 개수 (클수록 GPU 메모리 사용량 증가)
        args=training_args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir)

    checkpoint_output_dir = os.path.join(output_dir, 'deepseek_checkpoint')
    trainer.model.save_pretrained(checkpoint_output_dir)
    tokenizer.save_pretrained(checkpoint_output_dir)

    return lora_llm, tokenizer


# SFT 테스트를 위한 모델 로딩
# Create Date : 2025.03.21
# Last Update Date : 2025.03.22
# - Auto-GPTQ 적용

# Arguments:
# - 없음

# Returns:
# - llm       (LLM)       : SFT 로 Fine-tuning 된 LLM (없으면 None)
# - tokenizer (tokenizer) : 해당 LLM 에 대한 tokenizer (LLM 이 없으면 None)

def load_sft_llm():
    print('loading LLM ...')

    try:
        model = AutoModelForCausalLM.from_pretrained("sft_model", device_map='cuda')
        tokenizer = AutoTokenizer.from_pretrained("sft_model", eos_token='<eos>')
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    except Exception as e:
        print(f'loading LLM failed : {e}')
        return None, None


# SFT 로 Fine-Tuning 된 LLM 을 테스트
# Create Date : 2025.03.21
# Last Update Date : 2025.03.22
# - log 파일명 수정 (log_llm_test_result.csv -> log_test_sft_result.csv)

# Arguments:
# - llm              (LLM)        : SFT 로 Fine-tuning 된 LLM
# - tokenizer        (tokenizer)  : 해당 LLM 의 tokenizer
# - shape_infos      (list(dict)) : 해당 LLM 이 valid/test dataset 의 각 prompt 에 대해 생성해야 할 도형들에 대한 정보
# - llm_prompts      (list(str))  : 해당 LLM 에 전달할 User Prompt (Prompt Engineering 을 위해 추가한 부분 제외)
# - llm_dest_outputs (list(str))  : 해당 LLM 의 목표 output 답변

# Returns:
# - llm_answers (list(str)) : 해당 LLM 의 답변
# - final_score (float)     : 해당 LLM 의 성능 score
#
# - log/log_test_sft_result.csv 에 해당 LLM 테스트 기록 저장
# - 모델의 답변을 이용하여, sft_model_diagrams/diagram_{k}.png 다이어그램 파일 생성

def test_sft_llm(llm, tokenizer, shape_infos, llm_prompts, llm_dest_outputs):
    result_dict = {'prompt': [], 'answer': [], 'dest_output': [], 'time': [], 'score': []}

    for idx, (prompt, shape_info, dest_output) in enumerate(zip(llm_prompts, shape_infos, llm_dest_outputs)):
        inputs = tokenizer(f'### Question: {prompt}\n ### Answer: ',  return_tensors='pt').to(llm.device)

        with torch.no_grad():
            if idx == 0:
                print('llm output generating ...')

            start = time.time()
            outputs = llm.generate(**inputs, max_length=1536, do_sample=True)
            generate_time = time.time() - start
            llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace('<|EOT|>', '')
            llm_answer = llm_answer.split('### Answer: ')[1]  # prompt 부분을 제외한 answer 만 표시

        # llm answer 를 이용하여 Diagram 생성 및 저장
        try:
            llm_answer_lines = llm_answer.split('\n')
            os.makedirs(f'{PROJECT_DIR_PATH}/fine_tuning/sft_model_diagrams', exist_ok=True)
            diagram_save_path = f'{PROJECT_DIR_PATH}/fine_tuning/sft_model_diagrams/diagram_{idx:06d}.png'
            generate_diagram_from_lines(llm_answer_lines, diagram_save_path)

        except Exception as e:
            print(f'SFT diagram generation failed: {e}')

        score = compute_output_score(shape_info=shape_info, output_data=llm_answer)

        result_dict['prompt'].append(prompt)
        result_dict['answer'].append(llm_answer)
        result_dict['dest_output'].append(dest_output)
        result_dict['time'].append(generate_time)
        result_dict['score'].append(score)

        pd.DataFrame(result_dict).to_csv(f'{PROJECT_DIR_PATH}/fine_tuning/log/log_test_sft_result.csv')

    llm_answers = result_dict['answer']
    final_score = np.mean(result_dict['score'])

    return llm_answers, final_score


if __name__ == '__main__':

    # check cuda is available
    assert torch.cuda.is_available(), "CUDA MUST BE AVAILABLE"
    print(f'cuda is available with device {torch.cuda.get_device_name()}')

    sft_dataset_path = f'{PROJECT_DIR_PATH}/create_dataset/sft_dataset_llm.csv'
    df = pd.read_csv(sft_dataset_path)

    add_text_column_for_llm(df)  # LLM 이 학습할 수 있도록 text column 추가

    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=2025)

    # LLM Fine-tuning
    llm, tokenizer = load_sft_llm()

    if llm is None or tokenizer is None:
        print('LLM load failed, fine tuning ...')
        llm, tokenizer = run_fine_tuning(df_train, df_valid)
    else:
        print('LLM load successful!')

    # LLM 테스트
    print('LLM test start')

    llm_prompts = df_valid['input_data'].tolist()
    llm_dest_outputs = df_valid['output_data'].tolist()
    shape_infos = df_valid['dest_shape_info'].tolist()

    llm_answer, final_score = test_sft_llm(llm, tokenizer, shape_infos, llm_prompts, llm_dest_outputs)
    print(f'\nLLM FINAL Score :\n{final_score}')
```

</details>

### 4-1. GPTQ 적용만 했을 때

```optimum``` 라이브러리가 없다는 오류 발생

```
Traceback (most recent call last):
  File "fine_tuning/sft_fine_tuning.py", line 214, in <module>
    llm, tokenizer = run_fine_tuning(df_train, df_valid)
  File "fine_tuning/sft_fine_tuning.py", line 51, in run_fine_tuning
    original_llm = AutoModelForCausalLM.from_pretrained(model_path,
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\models\auto\auto_factory.py", line 564, in from_pretrained
    return model_class.from_pretrained(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\modeling_utils.py", line 3652, in from_pretrained
    hf_quantizer = AutoHfQuantizer.from_config(config.quantization_config, pre_quantized=pre_quantized)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\quantizers\auto.py", line 148, in from_config
    return target_cls(quantization_config, **kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\quantizers\quantizer_gptq.py", line 47, in __init__
    from optimum.gptq import GPTQQuantizer
ModuleNotFoundError: No module named 'optimum'
```

### 4-2. optimum 라이브러리 최신 버전 설치 후

* optimum 라이브러리의 최신 버전 (1.23.3) 설치
* 기존에 ```transformer==4.46.3```, ```auto-gptq==0.7.1``` 설치된 상태
* 아래와 같은 오류 발생

```
Traceback (most recent call last):
  File "fine_tuning/sft_fine_tuning.py", line 214, in <module>
    llm, tokenizer = run_fine_tuning(df_train, df_valid)
  File "fine_tuning/sft_fine_tuning.py", line 99, in run_fine_tuning
    trainer.train()
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\trl\trainer\sft_trainer.py", line 434, in train
    output = super().train(*args, **kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\trainer.py", line 2123, in train
    return inner_training_loop(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\trainer.py", line 2481, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\transformers\trainer.py", line 3612, in training_step
    self.accelerator.backward(loss, **kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\accelerate\accelerator.py", line 2246, in backward
    loss.backward(**kwargs)
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\_tensor.py", line 521, in backward
    torch.autograd.backward(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\autograd\__init__.py", line 289, in backward
    _engine_run_backward(
  File "C:\Users\20151\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\autograd\graph.py", line 768, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
  0%|          | 0/2240 [00:02<?, ?it/s] 
```

### 4-3. QLoRA 처리

* 아래와 같이 ```prepare_model_for_kbit_training``` 을 통해 **Quantize 된 모델로서 학습이 가능하도록 Original LLM 을 변환**
* 시도는 성공했으나, **학습 및 추론 시간의 향상이 체감되지 않음**

```python
quantized_llm = prepare_model_for_kbit_training(original_llm)
lora_llm = get_peft_model(quantized_llm, lora_config)
```

### 4-4. ```device_map='cuda'``` 추가

* 기존의 ```AutoModelForCausalLM.from_pretrained(...).cuda()``` 형태에서 ```device_map='cuda'``` 로 인수를 지정하는 것으로 형태 변경
* 학습 속도 향상 체감 안됨

```python
original_llm = AutoModelForCausalLM.from_pretrained(model_path,
                                                    device_map='cuda',
                                                    torch_dtype=torch.float16)
```