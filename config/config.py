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
        self.mode = "train"  # train,valid
        self.max_pooling_loss = False
        self.spectrogram = 'mel'  # mfcc,mel
        self.label_id = 0  # nihaolele,lele,whole
        self.label_list = ['nihaolele', 'lele', 'whole']
        self._num_classes = [2, 1, 1]

        self.model_path = './params/mel/'
        self.working_path = './graph/mel/'
        self.graph_name = 'graph.pb'
        self.data_path = './data/mel/'

        # self.model_path = './params/lele_mfcc/'
        # self.working_path = './params/lele_mfcc/'
        # self.data_path = './data/lele_mfcc/'

        # self.data_path = '/ssd/liuziqi/mel/'
        self.model_name = 'latest.ckpt'
        self.rawdata_path = './rawdata/'
        # self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'
        self.gpu = "0"

        self.fft_size = 400
        self.step_size = 160
        self.samplerate = 16000

        self.cell_clip = 3.
        self.num_layers = 1
        self.init_scale = 0.1
        self.learning_rate = 5e-5
        self.max_grad_norm = 5
        self.num_layers = 1
        self.num_features = 60
        self.hidden_size = 64
        self.use_project = False
        self.num_proj = 32
        self.max_epoch = 200
        self.keep_prob = 1.0
        self.grad_clip = -1
        self.lr_decay = 0.5
        self.batch_size = 32
        self.tfrecord_size = 32
        self.trigger_threshold = 0.7  # between (0,1), but this param is somehow arbitrary

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
        return self._num_classes[self.label_id] + 1

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
