# encoding: utf-8

'''

@author: ZiqiLiu


@file: customize.py

@time: 2017/7/24 下午1:42

@desc:
'''
import pickle
import requests
import os
import json
from utils.common import check_dir, path_join
from config.attention_config import get_config
from marker.pkg.pinyin_tone.pinyin import EngTonedMarker

config = get_config()

wav_customize_dir = config.rawdata_path + 'customize/' + config.custom_keyword
wav_customize_valid_dir = config.rawdata_path + 'customize_valid/' + config.custom_keyword


def collect_customize_data(valid=False):
    check_dir(wav_customize_valid_dir)
    check_dir(wav_customize_dir)

    marker = EngTonedMarker()

    device_id = '32EFEA3263D079E1BE3767C87FC0A1C2'
    base_url = 'http://speechreview.in.naturali.io/prod/'

    r = requests.get(base_url + 'get?limit=50&offset=0&deviceid=' + device_id)
    a = r.content
    j = json.loads(a.decode())

    records = j['Detail'][:3]
    pkl = []
    for record in records:
        download_url = base_url + 'audio?key=' + record['awskey']
        label = record['nires']
        print(download_url)
        wave = requests.get(download_url).content
        ret = marker.mark(label)
        print(ret)
        audio_name = record['awskey']
        if valid:
            path = path_join(wav_customize_valid_dir, audio_name)
        else:
            path = path_join(wav_customize_dir, audio_name)
        with open(path, 'wb') as f:
            f.write(wave)

        slow = path.replace('.wav', '_slow.wav')
        fast = path.replace('.wav', '_fast.wav')
        loud = path.replace('.wav', '_loud.wav')
        silent = path.replace('.wav', 'slient.wav')
        os.system(
            'ffmpeg -i "%s" -filter:a "atempo=0.8" -vn -y -loglevel 0 "%s"' % (
                path, slow))
        os.system(
            'ffmpeg -i "%s" -filter:a "atempo=1.2" -vn -y -loglevel 0 "%s"' % (
                path, fast))
        os.system(
            'ffmpeg -y -i "%s" -vn -sn -filter:a volume=2.0dB -ar 16000 -loglevel 0 "%s"' % (
                path, loud))
        print(path)
        print(loud)
        os.system(
            'ffmpeg -y -i "%s" -vn -sn -filter:a volume=-2.0dB -ar 16000 -loglevel 0 "%s"' % (
                path, silent))

        file_names = [audio_name, fast.split('/')[-1], slow.split('/')[-1],
                      loud.split('/')[-1], silent.split('/')[-1]]
        for name in file_names:
            if valid:
                pkl.append((name, 1, ret))
            else:
                pkl.append((name, ret, label))
    pkl.extend(pkl)
    pkl.extend(pkl[:2])
    if not valid:
        pkl.extend(pkl)
    for p in pkl:
        print(p)
    print(len(pkl))

    if valid:
        pkl_path = path_join(wav_customize_valid_dir, 'valid.pkl')
    else:
        pkl_path = path_join(wav_customize_dir, 'data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl, f)


if __name__ == '__main__':
    collect_customize_data(False)
