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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='train: train model, ' +
                                       'valid: model validation, ',
                        default=None)
    parser.add_argument('--model', help='rnn or attention',
                        default=None)
    parser.add_argument('-opt', '--optimizer',
                        help='optimzier:adam sgd nesterov',
                        default=None)
    parser.add_argument('--ktq',
                        help='whether run in ktq', type=int,
                        default=0)
    parser.add_argument('-clip', '--max_grad_norm', help='max_grad_norm clip',
                        type=float,
                        default=None)
    parser.add_argument('-epoch', '--max_epoch',
                        help='max_epoch',
                        type=int,
                        default=None)
    parser.add_argument('-reset', '--reset_global',
                        help='reset global step',
                        type=int,
                        default=None)
    parser.add_argument('-m', '--model_path',
                        help='The  model path for restoring',
                        default=None)
    parser.add_argument('-s', '--save_path',
                        help='The  model path for  saving',
                        default=None)
    parser.add_argument('-graph', '--graph_path',
                        help='The  path for saving graph proto',
                        default=None)
    parser.add_argument('-g', '--gpu', help='visible GPU',
                        default=None)
    parser.add_argument('-layer', '--num_layers',
                        help='number of RNN layer',
                        type=int, default=None)
    parser.add_argument('-head', '--multi_head_num',
                        help='multi_head_num',
                        type=int, default=None)
    parser.add_argument('-hidden', '--hidden_size',
                        help='number of hidden unit in attention layer',
                        type=int, default=None)
    parser.add_argument('-feed', '--feed_forward_inner_size',
                        help='number of feed_forward_inner_size',
                        type=int, default=None)
    parser.add_argument('-l', '--learning_rate', help='learning rate',
                        type=float, default=None)
    parser.add_argument('--decay_step', help='decay_step',
                        type=int, default=None)
    parser.add_argument('-decay', '--lr_decay', help='lr_decay',
                        type=float, default=None)
    parser.add_argument('-keep', '--keep_prob', help='keep_pro',
                        type=float, default=None)
    parser.add_argument('--train_path', help='train data path', default=None)
    parser.add_argument('--valid_path', help='valid data path', default=None)
    parser.add_argument('--noise_path', help='noise data path', default=None)
    parser.add_argument('--freq_size', help='mel feature num', type=int,
                        default=None)
    parser.add_argument('--fmin', help='min freq for mel filter', type=int,
                        default=None)
    parser.add_argument('--fmax', help='max freq for mel filter', type=int,
                        default=None)
    parser.add_argument('-relu', '--use_relu', help='use relu in fc layer',
                        default=None)
    parser.add_argument('--power', help='mel filter power:1 or 2', type=int,
                        default=None)
    parser.add_argument('-batch', '--batch_size', help='batch size', type=int,
                        default=None)
    parser.add_argument('-bg', '--use_bg_noise', help='use_bg_noise', type=int,
                        default=None)
    parser.add_argument('-wh', '--use_white_noise', help='use_white_noise',
                        type=int, default=None)
    parser.add_argument('-combine', '--combine_frame', help='combine frames',
                        type=int, default=None)
    parser.add_argument('--mfcc', help='mfcc', type=int, default=None)
    parser.add_argument('--n_mfcc', help='mfcc number', type=int, default=None)
    parser.add_argument('--n_mel', help='mel number', type=int, default=None)
    parser.add_argument('--pre_emphasis', help='pre_emphasis', type=int,
                        default=None)
    parser.add_argument('-var', '--variational_recurrent',
                        help='variational_recurrent', type=int,
                        default=None)
    parser.add_argument('-res', '--use_residual',
                        help='use residual wrapper', type=int,
                        default=None)
    parser.add_argument('--value_clip', help='nn outputs value clip', type=int,
                        default=None)
    parser.add_argument('-bgmax', '--bg_decay_max_db',
                        help='bg_decay_max_db', type=int,
                        default=None)
    parser.add_argument('-bgmin', '--bg_decay_min_db',
                        help='bg_decay_min_db', type=int,
                        default=None)

    flags = parser.parse_args().__dict__
    return flags


flags = get_args()
model = flags['model']
if model == 'rnn':
    config = rnn_config.get_config()

elif model == 'attention':
    config = attention_config.get_config()
else:
    raise Exception('model %s not defined!' % model)

del (flags['model'])

if not flags['ktq']:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags['gpu'])

del (flags['ktq'])
del (flags['gpu'])

print(flags)


def parse_args():
    for key in flags:
        if key == '_customize_dict' and flags[key]:
            custom_dict = {}
            custom_keyword = ''
            for i, word in enumerate(flags[key]):
                # 0 for space, 1,2,3 for ni hao le, 4 for other words
                custom_dict[word] = i + 5
                custom_keyword += word

            if not hasattr(config, '_customize_dict'):
                print(
                    "WARNING: Invalid override with attribute _customize_dict ")
            else:
                setattr(config, '_customize_dict', custom_dict)
            if not hasattr(config, 'custom_keyword'):
                print(
                    "WARNING: Invalid override with attribute custom_keyword ")
            else:
                setattr(config, 'custom_keyword', custom_keyword)
        if flags[key] is not None:
            if not hasattr(config, key):
                print("WARNING: Invalid override with attribute %s" % (key))
            else:
                setattr(config, key, flags[key])
    print(config.__dict__)
    return config, model


if __name__ == '__main__':
    parse_args()
