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
        self.max_pooling_standardize = True
        self.ktq = False
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
        self.data_path = './data/mel/'

        self.data_path = '/ssd/liuziqi/mel_all5x/'
        self.model_name = 'best.ckpt'
        self.rawdata_path = './rawdata/'
        self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'
        self.gpu = "0"

        self.fft_size = 400
        self.step_size = 160
        self.samplerate = 16000

        self.cell_clip = 3.
        self.num_layers = 1
        self.init_scale = 0.1
        self.learning_rate = 5e-4
        self.max_grad_norm = 5
        self.num_features = 60
        self.hidden_size = 64
        self.use_project = False
        self.num_proj = 32
        self.max_epoch = 200
        self.drop_out_input = -1
        self.drop_out_output = -1
        self.grad_clip = 0
        self.lr_decay = 0.9
        self.decay_step = 20000
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
