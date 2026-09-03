
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

# 7. attribute getattr

class UserProfile:
    def __init__(self):
        self.theme = "dark"
        self.language = "ko"

user = UserProfile()
setting_key = "font_size"  # 사용자나 설정에 따라 동적으로 요구되는 속성

# 속성이 없을 때를 대비해 hasattr과 if문을 번거롭게 사용해야 합니다.
if hasattr(user, setting_key):
    value = user.font_size
else:
    value = "14px"  # 기본값 설정

if hasattr(user, "font_size"):
    value = 1
else:
    value = 0  # 기본값 설정

if hasattr(user, "font_size"):
    test = "abc"
else:
    value = 0  # 기본값 설정

# 8. re anti-patterns

import re
text = "Hello 123 World\nNext Line\tTabbed"

# 1. re.compile
# 안티패턴: '\d'가 일반 문자열로 처리됨 (파이썬 버전에 따라 경고 발생)
# 올바른 표현: re.compile(r'\d+')
bad_compile = re.compile('\d+')

# 2. re.match
# 안티패턴: 문자열 시작부터 알파벳을 매칭하려 하나 '\w'에 이스케이프 누락 위험
# 올바른 표현: re.match(r'\w+', text)
bad_match = re.match("\w+", text)

# 3. re.search
# 안티패턴: 텍스트 중간의 공백과 단어 경계를 찾을 때 이스케이프 혼선 가능
# 올바른 표현: re.search(r'\s\w+', text)
bad_search = re.search('\s\w+', text)

# 4. re.fullmatch
# 안티패턴: 전체 문자열 검증 시 패턴이 길어질수록 백슬래시 해석 오류 확률 증가
# 올바른 표현: re.fullmatch(r'[\s\S]+', text)
bad_fullmatch = re.fullmatch("[\s\S]+", text)

# 5. re.findall
# 안티패턴: 모든 숫자를 찾으려 할 때 일반 문자열 사용
# 올바른 표현: re.findall(r'\d', text)
bad_findall = re.findall('\d', text)

# 6. re.finditer
# 안티패턴: 반복자(Iterator) 반환 시에도 패턴 해석 단계에서 문제 유발 가능
# 올바른 표현: re.finditer(r'\b\w+\b', text)
bad_finditer = re.finditer("\b\w+\b", text)

# 7. re.split
# 안티패턴: 줄바꿈이나 공백 기준으로 쪼갤 때 이스케이프 오작동 위험
# 올바른 표현: re.split(r'\s+', text)
bad_split = re.split('\s+', text)

# 8. re.sub
# 안티패턴: 치환 패턴과 대체 텍스트 모두에서 백슬래시 처리 문제 발생 가능
# 올바른 표현: re.sub(r'\d+', r'NUMBER', text)
bad_sub = re.sub("\d+", 'NUMBER', text)

# 9. f = lambda x: ... -> def f(x): return ...

import math

f = lambda x: pow(x, 2)
function = lambda x, y: x + y
distance = lambda x, y, z: math.sqrt(x*x + y*y + z*z)


def test_function(function1: callable, function2: callable, function3: callable):
    return function1(1) + function2(1, 2) + function3(1, 2, 3)


function_result = test_function(
    function1=lambda x: pow(x, 2),
    function2=lambda x, y: x + y,
    function3=lambda x, y, z: math.sqrt(x*x + y*y + z*z)
)
print(function_result)

# 10. prefix & suffix

test_str = 'abcdefgh'

if test_str[:2] == 'ab':
    print(1)
if test_str[:4] == 'abcd':
    print(2)
if test_str[-2:] == 'gh':
    print(3)
if test_str[-3:] == 'fgh':
    print(4)
if test_str[len(test_str)-1:] == 'h':
    print(5)
if test_str[len(test_str) - 2:] == 'gh':
    print(6)
