# encoding: utf-8

'''

@author: ZiqiLiu


@file: common.py

@time: 2017/5/18 上午11:06

@desc:
'''

# -*- coding:utf-8 -*-
# !/usr/bin/python

import sys

sys.path.append('../')
sys.dont_write_bytecode = True

import time
from functools import wraps
import os
from glob import glob
import numpy as np
import tensorflow as tf
import math


def describe(func):
    ''' wrap function,to add some descriptions for function and its running time
    '''

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + '...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(str(func.__name__ + ' in ' + str(end - start) + ' s'))
        return result

    return wrapper


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def path_join(path1, path2):
    if (path1.endswith('/')):
        if (path2.startswith('/')):
            return path1 + path2[1:]
        else:
            return path1 + path2

    else:
        if (path2.startswith(('/'))):
            return path1 + path2
        else:
            return path1 + '/' + path2


def dense2sparse(array_1d):
    indices = []
    values = []
    for i, v in enumerate(array_1d):
        if v == 0:
            continue
        indices.append(i)
        values.append(v)
    return indices, values, len(array_1d)


def sparse2dense(indices, values, len):
    dense = np.zeros(len, dtype=np.int32)
    for i, v in zip(indices, values):
        dense[i] = v

def increment_id(i,n):
    return str(i).zfill(n)
