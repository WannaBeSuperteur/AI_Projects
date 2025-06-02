try:
    from memory_mechanism.inference_sbert import run_inference, run_inference_each_example
    from memory_mechanism.load_sbert_model import load_pretrained_sbert_model
except:
    from llm.memory_mechanism.inference_sbert import run_inference, run_inference_each_example
    from llm.memory_mechanism.load_sbert_model import load_pretrained_sbert_model

import pandas as pd

import os
PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


# Memory Mechanism 학습된 모델을 이용하여, saved memory 중 가장 적절한 1개의 메모리를 반환 (단, Cos-similarity >= 0.6 인 것들만)
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# - sbert_model      (S-BERT Model) : 학습된 Sentence BERT 모델
# - user_prompt      (str)          : Oh-LoRA 에게 전달할 사용자 프롬프트
# - memory_file_name (str)          : 메모리 파일 (txt) 의 이름 (예: test.txt)
# - threshold        (float)        : minimum cosine similarity threshold (default: 0.6)
# - verbose          (bool)         : 각 memory item 에 대한 score 출력 여부

# Returns:
# - best_memory (str) : 메모리 파일에서 찾은 best memory

def pick_best_memory_item(sbert_model, user_prompt, memory_file_name='test.txt', threshold=0.6, verbose=False):
    memory_file_path = f'{PROJECT_DIR_PATH}/llm/memory_mechanism/saved_memory/{memory_file_name}'

    # read memory file
    memory_file = open(memory_file_path, 'r', encoding='UTF8')
    memory_file_lines = memory_file.readlines()
    memory_item_list = []

    # compute similarity scores for each memory item
    for line_idx, line in enumerate(memory_file_lines):
        if len(line.replace(' ', '')) < 3:
            continue

        memory_text = line.split('\n')[0]
        similarity_score = run_inference_each_example(sbert_model, memory_text, user_prompt)
        memory_item_list.append({'memory_text': memory_text, 'cos_sim': similarity_score})

        if verbose:
            print(f'line {line_idx} -> memory: {memory_text}, cosine similarity: {similarity_score:.4f}')

    # pick best memory item
    memory_item_list.sort(key=lambda x: x['cos_sim'], reverse=True)

    if len(memory_item_list) == 0:
        best_memory = ''
    elif memory_item_list[0]['cos_sim'] >= threshold:
        best_memory = memory_item_list[0]['memory_text']
    else:
        best_memory = ''

    return best_memory


# Memory Mechanism 학습된 모델 로딩
# Create Date : 2025.06.02
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - sbert_model (S-BERT Model) : 학습된 Sentence BERT 모델

def load_sbert_model():
    model_path = f'{PROJECT_DIR_PATH}/llm/models/memory_sbert/trained_sbert_model'
    sbert_model = load_pretrained_sbert_model(model_path)

    return sbert_model


if __name__ == '__main__':

    # load test dataset
    test_dataset_csv_path = f'{PROJECT_DIR_PATH}/llm/memory_mechanism/test_dataset.csv'
    test_dataset_df = pd.read_csv(test_dataset_csv_path)

    # load S-BERT Model
    sbert_model = load_sbert_model()
    print('S-BERT Model (for memory mechanism) - Load SUCCESSFUL! 👱‍♀️')

    # run inference on test dataset
    run_inference(sbert_model, test_dataset_df)

    # pick best memory by user prompt, from saved memory
    while True:
        test_user_prompt = input('\nInput user prompt (Ctrl+C to finish) : ')
        best_memory_item = pick_best_memory_item(sbert_model,
                                                 test_user_prompt,
                                                 memory_file_name='test.txt',
                                                 verbose=True)

        if best_memory_item == '':
            print(f'\nNO BEST MEMORY ITEM (cos-sim threshold : 0.6)')
        else:
            print(f'\nbest memory item : {best_memory_item}')
