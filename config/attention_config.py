# encoding: utf-8

'''

@author: ZiqiLiu


@file: attention_config.py

@time: 2017/5/18 上午11:18

@desc:
'''


def get_config():
    return Config()


class Config(object):
    def __init__(self):
        self.mode = "train"  # train,valid,build
        self.customize = 0  # 0 not  1 transform  2 restore from customize
        self.mfcc = False
        self.ktq = False
        self.spectrogram = 'mel'  # mfcc,mel
        self.label_dict = {'ni3': 1, 'hao3': 2,
                           'le4': 3}  # 0 for space 4 for other
        self._customize_dict = {'ping2': 5, 'guo3': 6}
        self._origin_label_seq = '1233'
        self._customize_label_seq = '45'

        self.model_path = './params/ctc2/'
        self.save_path = './params/ctc2/'
        self.graph_path = './graph/23w/'
        self.graph_name = 'graph.pb'

        self.train_path = '/ssd/liuziqi/ctc_23w/train/'
        self.valid_path = '/ssd/liuziqi/ctc_23w/valid/'
        self.noise_path = '/ssd/liuziqi/ctc_23w/noise/'
        self.custom_path = '/ssd/liuziqi/ctc_23w/custom/'
        self.custom_valid_path = '/ssd/liuziqi/ctc_23w/custom_valid/'
        self.model_name = 'latest.ckpt'
        self.rawdata_path = './rawdata/'
        self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'


        # training flags
        self.reset_global = 0
        self.batch_size = 16
        self.tfrecord_size = 32
        self.valid_steps = 320
        self.gpu = "0"
        self.warmup = False
        self.learning_rate = 1e-3
        self.max_epoch = 200
        self.valid_step = 320
        self.lr_decay = 0.9
        self.decay_step = 40000
        self.use_relu = True
        self.optimizer = 'adam'  # adam sgd nesterov

        # pre process flags
        self.fft_size = 400
        self.hop_size = 160
        self.samplerate = 16000
        self.max_sequence_length = 2000
        self.power = 1
        self.fmin = 300
        self.fmax = 8000
        self.n_mfcc = 20
        self.n_mel = 60
        self.pre_emphasis = False

        # noise flags
        self.use_white_noise = False
        self.use_bg_noise = True
        self.bg_noise_prob_raise = 1.05
        self.bg_decay_max_db = -6
        self.bg_decay_min_db = -20
        self.bg_noise_prob = 0.5

        # model params
        self.combine_frame = 2
        self.num_layers = 3
        self.max_grad_norm = -1
        self.feed_forward_inner_size = 512
        self.keep_prob = 0.9
        self.multi_head_num = 8
        self.hidden_size = 128

    @property
    def num_classes(self):
        # word+1 for background
        if self.customize == 2:
            return len(self.label_dict) + len(self._customize_dict) + 3
        else:
            return len(
                self.label_dict) + 3  # 0 for space 4 for others, 5 for ctc blank

    @property
    def num_customize(self):
        return len(self._customize_dict)

    @property
    def customize_dict(self):
        return dict(self.label_dict, **self._customize_dict)

    @property
    def get_dict(self):
        if self.customize:
            return self.customize_dict()
        else:
            return self.label_dict

    @property
    def label_seqs(self):
        if self.customize:
            return [self._origin_label_seq, self._customize_label_seq]
        else:
            return [self._origin_label_seq]

    @property
    def beam_size(self):
        return self.num_classes - 1

    @property
    def freq_size(self):
        return self.n_mfcc * 3 if self.mfcc else self.n_mel

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
