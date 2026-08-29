
import logging
import math  # unused import
import numpy as np
import os

unused_variable = 123
TEST_DUPLICATED = 10


def unused_function():
    common_value = 'dddd abbbcccccddddd'
    print('abcd')


logger = logging.getLogger()
logger.info("3335555")
a = 300

def abc():
    common_value = 'dddd abbbcccccddddd'


def similar_func1(similar: float, same: float, unused_arg: str):
    print(similar, same)
    for i in range(10):
        print('a')
        print('bb')


def similar_func2(similar_: int, same: int, unused_arg: str):
    print(similar_, same)
    for idx in range(20):
        print('a')
        print('bb')


def get_current_version():
    return 'v1.2.6'

next_level = get_current_version()


def check_single_responsibility():
    """Check if the code follows single responsibility rule and suggest improvement points."""
    pass


python_code_no = 300
python_code_num = 300

# print(111)
# if a > 3:
#     print('abcd')
