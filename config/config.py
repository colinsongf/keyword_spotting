# encoding: utf-8

'''

@author: ZiqiLiu


@file: config.py

@time: 2017/5/18 上午11:18

@desc:
'''


def get_config():
    return Config()


class Config(object):
    def __init__(self):
        self.is_training = True
        self.max_pooling_loss = True

        self.cell_clip = 3.
        self.num_layers = 1
        self.init_scale = 0.1
        self.learning_rate = 5e-5
        self.max_grad_norm = 5
        self.num_layers = 1
        self.num_classes = 4  # word+1 for background
        self.num_features = 20
        self.hidden_size = 64
        self.num_proj = 32
        self.max_epoch = 20
        self.keep_prob = 1.0
        self.grad_clip = -1
        self.lr_decay = 0.5
        self.batch_size = 16
        self.validation_size = 8
        self.trigger_threshold = 0.7  # between (0,1), but this param is somehow arbitrary

        # these three sizes are frames, which depend on STFT frame size
        self.smoothing_window = 31
        self.latency = 30
        self.word_interval = 50
        self.lockout = 50

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
