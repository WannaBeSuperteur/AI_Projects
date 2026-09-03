
# 1. unpacking

my_list = [10, 20, 30]

a = my_list[0]
b = my_list[1]
c = my_list[2]

my_list = [1, 2, 3, 4, 5]

first = my_list[0]
rest = my_list[1:]

if __name__ == '__main__':
    d = my_list[0]
    e = my_list[1]

    print('test')
    f = my_list[0]
    g = my_list[1:]

# 2. open file

f = open("data.txt", "r")
file = open("output.txt", "w")
open_test = open("numbers.txt", 'r', encoding='utf-8')

# 3. key=lambda -> itemgetter

test_array = []

my_list.sort(key=lambda x: x['key'])
my_list.sort(key=lambda x: x["key"])
test_array.sort(key=lambda x: x["chill_guy"])

key_name = 'chill_guy'
test_array.sort(key=lambda x: x[key_name])

# 4. check f-string

numeric = my_list[0]

a = 'dddd' + key_name
b = "dddddd" + key_name
c = key_name + '21345'
d = key_name + "13456"
e = "a" + "bb" + "ccc" + str(1234)
f = key_name + key_name
g = int('123') + numeric
h = numeric + (int("123"))

# 5. collections & itertools use
## 5-1. collections (빈도수)

words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
word_counts = {}

for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

word_counts = {}
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1

## 5-2. itertools.chain

matrix = [[1, 2], [3, 4], [5, 6]]

flattened = []
for row in matrix:
    for item in row:
        flattened.append(item)

my_list = [1, 2]
my_tuple = (3, 4)
my_set = {5, 6}

# 타입을 맞추기 위해 불필요한 변환이 일어남
combined = my_list + list(my_tuple) + list(my_set)
for item in combined:
    print(item)

dict_a = {'a': 1, 'b': 2}
dict_b = {'c': 3, 'd': 4}

# keys()나 values()를 리스트로 변환해 합치는 낭비 발생
all_values = list(dict_a.values()) + list(dict_b.values())
all_values_ = list(dict_a.values()) + my_list

## 5-3. glob.glob 사용 (파일 경로명 조건 리스트)

import os

target_dir = "./my_folder"
txt_files = []

for filename in os.listdir(target_dir):
    # 파일명 조건 검사 (문자열 매칭)
    if filename.endswith(".txt"):
        # 경로명을 다시 결합해야 하는 번거로움 발생
        full_path = os.path.join(target_dir, filename)
        txt_files.append(full_path)

print(txt_files)

# 6. func args bindable


def test_func(value_1: int, value_2: int, value_3: int) -> int:
    return value_1 + value_2 + value_3

def test_func2() -> int:
    return 0
