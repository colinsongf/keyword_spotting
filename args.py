# encoding: utf-8

'''

@author: ZiqiLiu


@file: args.py

@time: 2017/6/19 下午3:35

@desc:
'''
import argparse
import os
from config import attention_config, rnn_config


def config_value_cast(config, key, value):
    _type = type(getattr(config, key))
    if _type is bool:
        return value.lower() not in ['false', 'f', 'no', 'n', 'off', '0']
    else:
        return type(getattr(config, key))(value)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='train: train model, ' +
                                       'valid: model validation, ',
                        default=None)
    parser.add_argument('--model', help='rnn or attention',
                        default=None)
    parser.add_argument('--ktq',
                        help='whether run in ktq', type=int,
                        default=0)
    parser.add_argument('--train_path', help='train data path',
                        default='/ssd/keyword/ctc_23w/train/')
    parser.add_argument('--valid_path', help='valid data path',
                        default='/ssd/keyword/ctc_23w/valid/')
    parser.add_argument('--noise_path', help='noise data path',
                        default='/ssd/keyword/ctc_23w/noise/')
    parser.add_argument('-o', '--override', nargs='*', default=[],
                        help='Override configuration, with k-v pairs')

    flags = parser.parse_args().__dict__
    return flags


flags = get_args()
model = flags['model']
mode = flags['mode']
if model == 'rnn':
    config = rnn_config.get_config()

elif model == 'attention':
    config = attention_config.get_config()
else:
    raise Exception('model %s not defined!' % model)

if not flags['ktq']:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags['gpu'])

print(flags)


def parse_args():
    setattr(config, 'mode', mode)
    L = len(flags['override'])
    assert L % 2 == 0
    for i in range(L // 2):
        key, value = flags['override'][2 * i], flags['override'][2 * i + 1]
        if not hasattr(config, key):
            print("WARNING: Invalid override with attribute %s" % (key))
        else:
            setattr(config, key, config_value_cast(config, key, value))
    for key in ['train_path','valid_path','noise_path']:
        if not hasattr(config, key):
            print("WARNING: Invalid override with attribute %s" % (key))
        else:
            setattr(config, key, config_value_cast(config, key, value))


    return config, model, mode


if __name__ == '__main__':
    parse_args()
