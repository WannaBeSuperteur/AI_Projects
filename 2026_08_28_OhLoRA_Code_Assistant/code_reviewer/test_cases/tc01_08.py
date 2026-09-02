import math
import os

import numpy as np
import pandas
import sys
from collections import abc

import tc01_02
import re
from torch import nn


print('ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd more than line length 120 =========================')


def without_type_hint(a, b: int, c: str):
    return a + b + len(c)


def with_type_hint(a: int, b: int, c: str) -> int:
    return a + b + len(c)


print(without_type_hint(2025, 113, 'REBEL HEART'))
print(with_type_hint(2025, 113, 'REBEL HEART'))


def long_function_with_docstring():
    """This is a docstring."""

    I, J, K, L = 2, 3, 4, 5

    for i in range(2):
        for j in range(3):
            for k in range(4):
                for l in range(5):
                    print(i)
                    print(j)
                    print(k)
                    print(l)

                print('aaaa',
                      'bbbb',
                      'cccc')

    print(I, J, K, L)


print(1234)


def long_function_wo_docstring():
    I, J, K, L = 2, 3, 4, 5

    for i in range(2):
        for j in range(3):
            print('aaaaaaaaaaaa',
                  'bbbbbbb',
                  'cccc')

            test_dict_test_dict = {'aaa': 1,
                                   'bbb': 2,
                                   'ccc': 3}
            print(test_dict_test_dict)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                print('aaaaaaaaaaaa',
                      'bbbbbbb',
                      'cccc')

                print('x')

                test_dict_test_dict = {'aaa': 1,
                                       'bbb': 2,
                                       'ccc': 3}
                print(test_dict_test_dict)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                for l in range(5):
                    print(i)
                    print(j)

                    print(k)
                    print(l)

    print(I, J, K, L)


print(5678)


def too_long_function():
    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)

    print(1)
    print(2)
    for i in range(10):
        print(3)


too_long_function()
