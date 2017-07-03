# encoding: utf-8

'''

@author: ZiqiLiu


@file: review_spider.py

@time: 2017/6/9 下午5:06

@desc:
'''
import requests
import json
import os
import re
import pickle
from glob import glob

base_url = 'http://speechreview.in.naturali.io/prod/'
i = 0
p = re.compile(r'[a-zA-Z]+')
wave_list = []
exist = set()
if os.path.exists('./download/list.pkl'):
    with open('./download/list.pkl', 'rb') as f:
        wav_list = pickle.load(f)
    exist = set([i[1] for i in wave_list])

try:
    while True:
        print(i)
        r = requests.get(
            base_url + 'get?limit=50&offset=%d&maxid=16335321' % (i * 50))

        i += 1
        a = r.content
        records = json.loads(a.decode())['Detail']
        for r in records:

            key = r['nires']
            wav_file = r['awskey']
            if not wav_file in exist and len(key) > 1 and len(key) < 12:
                if ('你' in key or '好' in key or (
                                '乐' in key and not '音乐' in key)) and not p.findall(
                    key):
                    download_url = base_url + 'audio/' + wav_file
                    wave = requests.get(download_url).content
                    wave_list.append((wav_file, key))
                    path = "./download/" + wav_file
                    print(r['awskey'])
                    with open(path, 'wb') as f:
                        f.write(wave)

                        # if key.count('乐乐') == 1 and key.count('你好') == 0:
                        #     download_url = base_url + 'audio/' + key
                        #     wave = requests.get(download_url).content
                        #     path = "./lele/" + key
                        #     print(r['awskey'])
                        #     with open(path, 'wb') as f:
                        #         f.write(wave)
                        # elif key.count('你好') == 1 and key.count('乐乐') == 1:
                        #     download_url = base_url + 'audio/' + key
                        #     wave = requests.get(download_url).content
                        #     path = "./nihaolele/" + key
                        #     print(r['awskey'])
                        #     with open(path, 'wb') as f:
                        #         f.write(wave)
except KeyboardInterrupt:
    with open('./download/list.pkl', 'wb') as f:
        pickle.dump(wave_list, f)
    wave_list = sorted(wave_list, key=lambda k: k[0])
    new_list = []
    current = None
    for i in wave_list:
        if i[0] == current:
            continue
        else:
            current = i[0]
            new_list.append(i)
    with open('./download/new_list.pkl', 'wb') as f:
        pickle.dump(new_list, f)
