from trl import DataCollatorForCompletionOnlyLM
from copy import deepcopy
import random


augment_lists = [[' 나는 ', ' 난 '],
                 ['나에게 ', '내게 '],
                 ['인공지능 ', 'AI '],
                 ['인공지능으로', 'AI로'],
                 ['음…', '음,', '흠…', '흠,'],
                 ['오!', '우와!', '와우!', '와우', '우와'],
                 ['ㅠㅠ', 'ㅜㅜ', 'ㅠ'],
                 ['ENTJ', '엔티제'],
                 ['LLM', '거대 언어 모델', '대형 언어 모델', '대규모 언어 모델'],
                 ['음악 ', '노래 '],
                 ['음악은', '노래는'],
                 ['축하해', '축하축하', 'ㅊㅋㅊㅋ']]


class AugmentCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, response_template, llm_name, **kwargs):
        super().__init__(response_template, **kwargs)
        self.tokenizer = kwargs['tokenizer']
        self.llm_name = llm_name

    def augment(self, text):
        decoded_text = self.tokenizer.decode(text)
        augmented_text = decoded_text

        for augment_list in augment_lists:
            augment_list_shuffled = deepcopy(augment_list)
            random.shuffle(augment_list_shuffled)

            for word in augment_list_shuffled:
                if word in augmented_text:
                    replaced_word = random.choice(augment_list_shuffled)
                    augmented_text = augmented_text.replace(word, replaced_word)

        encoded_augmented_text = self.tokenizer.encode(augmented_text)

        if self.llm_name == 'kanana':
            return encoded_augmented_text[1:]
        else:
            return encoded_augmented_text

    def __call__(self, data_collection):
        for data in data_collection:
            data['input_ids'] = self.augment(data['input_ids'])
            data['attention_mask'] = [1 for _ in range(len(data['input_ids']))]

        return super().__call__(data_collection)
