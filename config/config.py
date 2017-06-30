# encoding: utf-8

'''

@author: ZiqiLiu


@file: config.py

@time: 2017/5/18 上午11:18

@desc:
'''
import argparse


def get_config():
    return Config()


class Config(object):
    def __init__(self):
        self.mode = "train"  # train,valid,build
        self.ktq = False
        self.spectrogram = 'mel'  # mfcc,mel
        self.label_dict = {'ni3': 1, 'hao3': 2, 'le4': 3}  # 0 for space 4 for other

        self.reset_global = 0

        self.model_path = './params/ctc1/'
        self.save_path = './params/ctc1/'
        self.graph_path = './graph/mel/'
        self.graph_name = 'graph.pb'

        self.train_path = '/ssd/liuziqi/ctc_pinyin/train/'
        self.valid_path = '/ssd/liuziqi/ctc_pinyin/valid/'
        self.model_name = 'best.ckpt'
        self.rawdata_path = './rawdata/'
        self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'
        self.gpu = "0"

        self.fft_size = 400
        self.step_size = 160
        self.samplerate = 16000
        self.power = 1
        self.fmin = 300
        self.fmax = 8000

        self.warmup = True
        self.learning_rate = 1.5e-3
        self.max_epoch = 200
        self.valid_step = 320
        self.lr_decay = 0.9
        self.decay_step = 40000
        self.batch_size = 16
        self.tfrecord_size = 32
        self.use_relu = True
        self.use_white_noise = False
        self.optimizer = 'adam'  # adam sgd nesterov

        self.cell_clip = 3.
        self.num_layers = 2
        self.init_scale = 0.1
        self.max_grad_norm = -1
        self.freq_size = 60
        self.feed_forward_inner_size = 512
        self.keep_prob = 0.9
        self.multi_head_num = 8
        self.model_size = 128
        self.use_project = False
        self.num_proj = 32




    @property
    def num_classes(self):
        # word+1 for background
        return len(self.label_dict)+3  # 0 for space 4 for others, 5 for ctc blank

    @property
    def beam_size(self):
        return self.num_classes-1

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
