# encoding: utf-8

'''

@author: ZiqiLiu


@file: download_filter.py

@time: 2017/7/2 下午9:32

@desc:
'''
import pickle
import librosa
import os
from tqdm import tqdm

with open('./download/list.pkl', 'rb') as f:
    wave_list = pickle.load(f)
new_list = []

max_len = 16000 * 12
for i in tqdm(wave_list):
    y, sr = librosa.load('./download/' + i[0], sr=16000)
    if len(y) > max_len:
        os.system('rm ./download/%s' % i[0])
    else:
        new_list.append(i)

with open('./download/list.pkl.filtered', 'wb') as f:
    pickle.dump(new_list, f)
