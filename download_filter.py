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
from glob import glob

max_len = 16000 * 12
files = glob('./download/*.wav')
wave_list = []
for f in files:
    y, sr = librosa.load(f, sr=16000)
    if len(y) > max_len:
        os.system('rm %s' % f)
    else:
        name = f.split('/')[-1]
        label = name.split('_')[-1][:-4]
        wave_list.append((name, label))
print(len(wave_list))
with open('./download/0706.pkl', 'wb') as f:
    pickle.dump(wave_list, f)
