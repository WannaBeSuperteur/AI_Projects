
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

primes = arr
print(sum([p for p in primes if p >= 10]))
print(max([p for p in primes if p % 3 == 2]))
print(min([p for p in primes if p % 3 == 1]))
print(all([p % 2 == 1 for p in primes]))
print(any([p % 2 == 1 for p in primes]))

print(sum(p for p in primes if p >= 10))
print(max(p for p in primes if p % 3 == 2))
print(min(p for p in primes if p % 3 == 1))
print(all(p % 2 == 1 for p in primes))
print(any(p % 2 == 1 for p in primes))

test = 0
if test >= 100:
    print('large', 'S')
elif test >= 75:
    print('medium', 'A')
elif test >= 50:
    print('small', 'B')
else:
    print('tiny', 'C')

test_abc = 0
if test_abc >= 100:
    print('large', 'S')
elif test >= 75:
    print('medium', 'A')
elif test_abc >= 50:
    print('small', 'B')
else:
    print('tiny', 'C')

test_def = 0
if test_def >= 100:
    print('large', 'S')
elif test_def >= 75:
    print('medium', 'A')
elif test_def >= 50:
    print('small', 'B', 'passed')
else:
    print('tiny', 'C')

if test >= 100 and True:
    print('aaa')
elif test <= 75 and False:
    print('bbb')
else:
    print('ccc')
