# encoding: utf-8

'''

@author: ZiqiLiu


@file: fetch_wave.py

@time: 2017/6/9 下午5:06

@desc:
'''
import requests
import json

device_id = '8FB56F7E4981B8D13D279C3C9BE5DEC5'
base_url = 'http://speechreview.in.naturali.io/prod/'
headers = {
    'Cookie': '_oauth2_proxy=bGl1enExMkBob3RtYWlsLmNvbQ==|1500630961|RuOszsSPudg9snnxfCTLpnvSLHQ='}


def fetch(id):
    url = base_url + 'get?limit=50&offset=0&deviceid=' + id
    try:

        r = requests.get(url)
        a = r.content
        j = json.loads(a.decode())

        record = j['Detail'][0]
        download_url = base_url + 'audio?key=' + record['awskey']
        label = record['nires']
        print(download_url)
        wave = requests.get(download_url).content
        # print(wave)
        path = "./temp.wav"
        with open(path, 'wb') as f:
            f.write(wave)
        return path, label, record['awskey']
    except Exception as e:
        print(url)