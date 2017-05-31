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
        self.mode = "valid"  # train,valid
        self.max_pooling_loss = False

        self.model_path = './params//mel20/'
        self.working_path = './params/mel20/'
        self.data_path = './data/mel20/'
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
        self.num_classes = 3  # word+1 for background
        self.num_features = 20
        self.hidden_size = 64
        self.num_proj = 32
        self.max_epoch = 200
        self.keep_prob = 1.0
        self.grad_clip = -1
        self.lr_decay = 0.5
        self.batch_size = 64
        self.validation_size = 32
        self.trigger_threshold = 0.5  # between (0,1), but this param is somehow arbitrary

        # these three sizes are frames, which depend on STFT frame size
        self.smoothing_window = 31
        self.latency = 30
        self.word_interval = 50
        self.lockout = 50

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
