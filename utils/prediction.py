# encoding: utf-8

'''

@author: ZiqiLiu


@file: prediction.py

@time: 2017/6/14 下午4:30

@desc:
'''

import numpy as np


def predict(moving_avg, threshold, lockout, f=None):
    if f is not None:
        print(f)
    # array is 2D array, moving avg for one record, whose shape is (t,p)
    # label is one-hot, which is also the same size of moving_avg
    # we assume label[0] is background
    # return a trigger array, the same size (t,p) (it's sth like mask)
    # print('=' * 50)
    num_class = moving_avg.shape[1]
    len_frame = moving_avg.shape[0]
    # print(num_class, len_frame)
    prediction = np.zeros(moving_avg.shape, dtype=np.float32)
    for i in range(1, num_class):
        j = 0
        while j < len_frame:
            if moving_avg[j][i] > threshold:
                prediction[j][i] = 1
            j += 1
    return prediction


def decode(prediction, word_interval, golden):
    raw = golden
    keyword = list(raw)
    # prediction based on moving_avg,shape(t,p),sth like one-hot, but can may overlapping
    # prediction = prediction[:, 1:]
    num_class = prediction.shape[1]
    len_frame = prediction.shape[0]
    pre = 0
    inter = 0

    target = keyword.pop()
    # try:
    for frame in prediction:
        if frame.sum() > 0:
            if pre == 0:
                assert frame.sum() == 1
                index = np.nonzero(frame)[0]
                pre = index[0]
                if index == target:
                    if inter < word_interval:
                        if len(keyword) == 0:
                            return 1
                        target = keyword.pop()
                        continue
                keyword = list(raw)
                target = keyword.pop()
                inter = 0
                if index == target:
                    target = keyword.pop()
                    continue
            else:
                if frame[pre] == 1:
                    continue
                else:
                    index = np.nonzero(frame)[0]
                    pre = index
                    if index == target:
                        if len(keyword) == 0:
                            return 1
                        target = keyword.pop()
                    else:
                        keyword = list(raw)
                        target = keyword.pop()
                        if index == target:
                            if len(keyword) == 0:
                                return 1
                            target = keyword.pop()
                            continue
        else:
            if pre == 0:
                if len(raw) - len(keyword) > 1:
                    inter += 1
            else:
                pre = 0

    return 0


def evaluate(result, target):
    assert len(result) == len(target)
    print(target)
    print(result)
    xor = [a ^ b for a, b in zip(target, result)]
    miss = sum([a & b for a, b in zip(xor, target)])
    false_accept = sum([a & b for a, b in zip(xor, result)])
    return miss, sum(target), false_accept


def moving_average(array, n=5, padding=True):
    # array is 2D array, logits for one record, shape (t,p)
    # return shape (t,p)
    if n % 2 == 0:
        raise Exception('n must be odd')
    if len(array.shape) != 2:
        raise Exception('must be 2-D array.')
    if n > array.shape[0]:
        raise Exception(
            'n larger than array length. the shape:' + str(array.shape))
    if padding:
        pad_num = n // 2
        array = np.pad(array=array, pad_width=((pad_num, pad_num), (0, 0)),
                       mode='constant', constant_values=0)
    array = np.asarray([np.sum(array[i:i + n, :], axis=0) for i in
                        range(len(array) - 2 * pad_num)]) / n
    return array
