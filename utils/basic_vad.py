# encoding: utf-8

'''

@author: ZiqiLiu


@file: basic_vad.py

@time: 2017/7/18 上午10:46

@desc:
'''
import numpy as np


def vad(sig, thres=40):
    return np.abs(sig).sum() > thres
