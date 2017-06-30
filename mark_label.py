# encoding: utf-8

'''

@author: ZiqiLiu


@file: cut.py

@time: 2017/6/9 下午1:19

@desc:
'''
import pickle
from marker.pkg.pinyin_tone.pinyin import EngTonedMarker

count = 0
new = []
marker = EngTonedMarker()
#
# with open('ctc_label.pkl', 'rb') as f:
#     a = pickle.load(f)
#     for i in a:
#         ret = marker.mark(i[1])
#         new.append((i[0], ret))
#         print(ret)
#
# with open('ctc_label_pinyin.pkl', 'wb') as f:
#     pickle.dump(new, f)


with open('ctc_valid.pkl', 'rb') as f:
    a = pickle.load(f)
    for i in a:
        ret = marker.mark(i[2])
        new.append((i[0], i[1], ret))
        print(ret)

with open('ctc_valid_pinyin.pkl', 'wb') as f:
    pickle.dump(new, f)
