
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
