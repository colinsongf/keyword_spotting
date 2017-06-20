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
    parser.add_argument('-max', '--max_pooling_loss', help='1: maxpooling, ' +
                                                           '0: cross entropy,',
                        type=int,
                        default=None)
    parser.add_argument('-clip', '--max_grad_norm', help='max_grad_norm clip',
                        type=int,
                        default=None)
    parser.add_argument('-st', '--max_pooling_standardize',
                        help='whether use maxpooling standardize',
                        type=int,
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
    parser.add_argument('-p', '--use_project',
                        help='whether to use projection in LSTM, 1 or 0',
                        type=int, default=None)
    parser.add_argument('-layer', '--num_layers',
                        help='number of RNN layer',
                        type=int, default=None)
    parser.add_argument('-l', '--learning_rate', help='learning rate',
                        type=float, default=None)
    parser.add_argument('--decay_step', help='decay_step',
                        type=int, default=None)
    parser.add_argument('-decay', '--lr_decay', help='lr_decay',
                        type=float, default=None)
    parser.add_argument('--drop_out_input', help='drop_out_input',
                        type=float, default=None)
    parser.add_argument('--drop_out_output', help='drop_out_output',
                        type=float, default=None)
    parser.add_argument('-label', '--label_id', help='label id',
                        type=int, default=None)
    parser.add_argument('-thres', '--threshold', help='threshold for trigger',
                        type=float, default=None)
    parser.add_argument('--data_path', help='data path', default=None)
    parser.add_argument('--feature_num', help='data path', type=int,
                        default=None)

    flags = parser.parse_args().__dict__
    return flags
