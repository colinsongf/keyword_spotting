# encoding: utf-8

'''

@author: ZiqiLiu


@file: args.py

@time: 2017/6/19 下午3:35

@desc:
'''
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='train: train model, ' +
                                       'valid: model validation, ',
                        default=None)
    parser.add_argument('-opt', '--optimizer',
                        help='optimzier:adam sgd nesterov',
                        default=None)
    parser.add_argument('-ktq', '--ktq',
                        help='whether run in ktq', type=int,
                        default=None)
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
    parser.add_argument('-warm', '--warmup',
                        help='lr warmup',
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
    parser.add_argument('-hidden', '--model_size',
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

    flags = parser.parse_args().__dict__
    return flags
