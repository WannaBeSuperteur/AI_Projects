
import logging
import math  # unused import
import numpy as np
import os

unused_variable = 123
TEST_DUPLICATED = 10
IMPORT_NUM = 0


def unused_function():
    common_value = 'dddd abbbcccccddddd'
    print('abcd')


logger = logging.getLogger()
logger.info("3335555")
a = 300


def abc():
    common_value = 'dddd abbbcccccddddd'


nparray = np.array([0.1, 0.2, 0.3])
nparray2 = np.array([0.1, 0.2, 0.3])


def similar_func1(similar: float, same: float, unused_arg: str):
    print(similar, same)
    print("Let's start!")

    for i in range(10):
        print('a')
        print('bb')


def similar_func2(similar_: int, same: int, unused_arg: str):
    print(similar_, same)
    print("Let's start!")

    for idx in range(20):
        print('a')
        print('bb')


def get_current_version():
    return 'v1.2.6'


default_version = 'v1.0.0'


def get_current_ver(version=default_version):
    return version


next_level = get_current_version()


def check_single_responsibility():
    """Check if the code follows single responsibility rule and suggest improvement points."""
    pass


python_code_no = 300
python_code_num = 300
test = python_code_no + python_code_num


def imported_function():
    print('do not use anti-patterns')


imported_value = 'import me'
imported_but_unused_value = 'why do you want to import me'


# print(111)
# if a > 3:
#     print('abcd')
