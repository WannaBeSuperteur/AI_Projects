TEST_DUPLICATED = 10
TEST_STR = 'dddd abbbcccccddddd'

from tc01_01_basics import IMPORT_NUM, imported_function, imported_value, imported_but_unused_value


def similar_func3(similar_: float, same: str, unused_arg: str):
    print(similar_, same)
    print("Let's start!")

    for idx in range(30):
        print('a')
        print('bb')


def get_current_version():
    return 'v1.2.7'


print(IMPORT_NUM)
imported_function()
print(imported_value)
