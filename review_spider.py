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
import json
from concurrent.futures import ThreadPoolExecutor

base_url = 'http://speechreview.in.naturali.io/prod/'

wave_list = {}

exist = set()
if os.path.exists('./download/list.pkl'):
    with open('./download/list.pkl', 'rb') as f:
        wav_list = pickle.load(f)
    exist = set(wave_list.keys())


def fetch():
    json_list = ['./dump/0630.json', './dump/0629.json']
    records = []
    for j in json_list:
        with open(j, 'r', encoding='utf-8') as f:
            records.extend(json.load(f)['Detail'])

    p = re.compile(r'[a-zA-Z]+')

    for r in records:

        key = r['nires']
        wav_file = r['awskey']
        queryid = r['queryid']
        deviceid = r['deviceid']

        if not wav_file in exist and len(key) > 1 and len(key) < 12:
            if ('你' in key or '好' in key or (
                            '乐' in key and not '音乐' in key)) and not p.findall(
                key):
                wave_list[wav_file] = (key, queryid, deviceid)
    print(len(wave_list))
    with open('./download/list.pkl', 'wb') as f:
        pickle.dump(wave_list, f)


def download(wave_dict=wave_list):
    model1 = 'old-model'
    model2 = 'mix-plus-full-type-decoder-0629'
    sbs_url = 'http://sbs-dev.naturali.io/api/asr/%s/prod/%s?deviceid=%s&contact=0&appinfo=0'

    if len(wave_dict) == 0:
        with open('./download/list.pkl', 'rb') as f:
            wave_dict = pickle.load(f)
    wave_list = [tuple([key] + list(wave_dict[key])) for key in wave_dict]

    def worker(wav_file, key, queryid, deviceid):

        q1 = \
            json.loads(
                requests.get(sbs_url % (model1, queryid, deviceid)).text,
                encoding='utf-8')[
                'result']
        q2 = \
            json.loads(
                requests.get(sbs_url % (model2, queryid, deviceid)).text,
                encoding='utf-8')[
                'result']
        if q1 == q2:
            download_url = base_url + 'audio/' + wav_file
            wave = requests.get(download_url).content
            wave_list.append((wav_file, key))
            path = "./download/" + wav_file
            print(key)
            with open(path, 'wb') as f:
                f.write(wave)
        else:
            print('skip')

    ex = ThreadPoolExecutor(max_workers=40)
    for wav_file, key, queryid, deviceid in wave_list:
        ex.submit(worker, wav_file, key, queryid, deviceid)

    print('done!')


def div_list(l, n):
    length = len(l)
    t = length // n
    quaters = [t * i for i in range(0, n)]
    ran = range(0, n - 1)
    result = [l[quaters[i]:quaters[i + 1]] for i in ran]
    result.append(l[quaters[n - 1]:len(l)])
    return result


if __name__ == '__main__':
    # fetch()
    download()
