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
        self.max_pooling_loss = False
        self.max_pooling_standardize = True
        self.ktq = False
        self.use_relu = True
        self.use_white_noise = True
        self.optimizer = 'adam'  # adam sgd nesterov
        self.spectrogram = 'mel'  # mfcc,mel
        self.label_id = 0  # nihaolele,lele,whole
        self.label_list = ['nihaolele', 'lele', 'whole']
        self._num_classes = [3, 2, 2]  # word+1 for background
        self._golden = [[2, 1], [1], [1]]

        self.reset_global = 0

        self.model_path = './params/mel_all/'
        self.save_path = './params/mel_all/'
        self.graph_path = './graph/mel/'
        self.graph_name = 'graph.pb'
        self.train_path = './data/mel/train'
        self.valid_path = './data/mel/valid'

        # self.train_path = '/ssd/liuziqi/mel_all5x_energy/train'
        # self.valid_path = '/ssd/liuziqi/mel_all5x_energy/valid'
        self.model_name = 'best.ckpt'
        self.rawdata_path = './rawdata/'
        self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'
        self.gpu = "0"

        self.fft_size = 400
        self.step_size = 160
        self.samplerate = 16000

        self.cell_clip = 3.
        self.num_layers = 2
        self.init_scale = 0.1
        self.learning_rate = 5e-3
        self.max_grad_norm = -1
        self.freq_size = 60
        self.feed_forward_inner_size = 512
        self.keep_prob = 0.9
        self.multi_head_num = 8
        self.model_size = 128
        self.use_project = False
        self.num_proj = 32
        self.max_epoch = 200
        self.valid_step = 320
        self.lr_decay = 0.8
        self.decay_step = 10000
        self.batch_size = 32
        self.tfrecord_size = 32
        self.trigger_threshold = 0.6  # between (0,1), but this param is somehow arbitrary

        # these three sizes are frames, which depend on STFT frame size
        self.smoothing_window = 9
        self.latency = 30
        self.word_interval = 70
        self.lockout = 50

    @property
    def label(self):
        return self.label_list[self.label_id]

    @property
    def num_classes(self):
        # word+1 for background
        return self._num_classes[self.label_id]

    @property
    def golden(self):
        return self._golden[self.label_id]

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
