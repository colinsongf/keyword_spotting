# encoding: utf-8

'''

@author: ZiqiLiu


@file: queue.py

@time: 2017/7/17 下午5:21

@desc:
'''


class SimpleQueue(object):
    def __init__(self, maxLen):
        self.content = []
        self.maxLen = maxLen
        self.len = 0

    def clear(self):
        self.content = []
        self.len = 0

    def add(self, item):
        if not self.full():
            self.content.append(item)
            self.len += 1
        else:
            del (self.content[0])
            self.content.append(item)

    def full(self):
        return self.len == self.maxLen

    def get_all(self):
        return self.content
