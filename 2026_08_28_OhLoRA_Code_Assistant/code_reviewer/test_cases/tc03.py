
arr = [2, 3, 5, 7, 11, 13, 17, 19]
new_arr = []

for item in arr:
    new_arr.append(item + 1)

for item in arr:
    if item % 4 == 1:
        new_arr.append(item)

result = ''
for item in arr:
    result += str(item) + ','

for item in arr:
    new_arr.append(1)

for item in arr:
    if item % 4 == 1:
        new_arr.append(2)

result = ''
for item in arr:
    result += str(3) + ','
