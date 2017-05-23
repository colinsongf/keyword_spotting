# encoding: utf-8

'''

@author: ZiqiLiu


@file: input_data.py

@time: 2017/5/11 下午12:33

@desc:
'''
import numpy as np
from tensorflow.python.framework import dtypes
from config.config import get_config
import pickle

config = get_config()
save_train_dir = './data/train/'
save_valid_dir = './data/valid/'


def dense_to_ont_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, train_wave, train_label, train_seqLength, valid_wave, valid_label, valid_seqLength, valid_name):
        self.wave = train_wave
        self.label = train_label
        self.seqLength = train_seqLength

        self.vi_wave = valid_wave
        self.vi_label = valid_label
        self.vi_seqLength = valid_seqLength
        self.valid_name = valid_name

        self.valid_name_dict = {}
        for i, name in enumerate(valid_name):
            self.valid_name_dict[name] = i
        if len(valid_wave) < config.batch_size:
            pad_num = config.batch_size - len(valid_wave)
            # self.vi_wave = np.pad(self.vi_wave, pad_width=((0, pad_num), (0, 0), (0, 0)), mode='constant',
            #                       constant_values=0)
            # self.vi_label = np.pad(self.vi_label, pad_width=((0, pad_num), (0, 0)), mode='constant', constant_values=0)
            # self.vi_seqLength = np.pad(self.vi_seqLength, pad_width=((0, pad_num)), mode='constant', constant_values=0)
            self.vi_wave = self.padding(self.vi_wave, config.batch_size)
            self.vi_label = self.padding(self.vi_label, config.batch_size)
            self.vi_seqLength = self.padding(self.vi_seqLength, config.batch_size)
            self.valid_name += [''] * (config.batch_size - len(valid_wave))

            # print(self.vi_wave.shape)
            # print(self.vi_label.shape)
            # print(self.vi_seqLength.shape)
            # print(len(self.valid_name))

    def padding(self, array, target_size):
        pad_num = target_size - len(array)
        dim = len(array.shape) - 1
        return np.pad(array, pad_width=((0, pad_num),) + ((0, 0),) * dim, mode='constant', constant_values=0)

    def test_data(self, name=None):
        if name is not None:
            index = self.valid_name_dict[name]
            vi_wave = np.zeros(self.vi_wave.shape, dtype=np.float32)
            vi_label = np.zeros(self.vi_label.shape, dtype=np.float32)
            vi_seq = np.zeros(self.vi_seqLength.shape, dtype=np.float32)
            vi_wave[0] = self.vi_wave[index]
            vi_label[0] = self.vi_label[index]
            vi_seq[0] = self.vi_seqLength[index]
            return vi_wave, vi_label, vi_seq, self.valid_name
        else:
            return self.vi_wave, self.vi_label, self.vi_seqLength, self.valid_name

    def next_batch(self, batch_size, shuffle=True):
        perm = np.arange(batch_size)
        np.random.shuffle(perm)
        return self.wave, self.label, self.seqLength


def read_dataset(dtype=dtypes.float32):
    train_wave = np.load(save_train_dir + 'wave.npy')
    train_label = np.load(save_train_dir + 'labels.npy')
    train_seqLen = np.load(save_train_dir + 'seqLen.npy')

    valid_wave = np.load(save_valid_dir + 'wave.npy')
    valid_label = np.load(save_valid_dir + 'labels.npy')
    valid_seqLen = np.load(save_valid_dir + 'seqLen.npy')
    with open(save_valid_dir + 'filename.pkl', 'rb') as fs:
        valid_name = pickle.load(fs)
    # print(wave.shape)
    # print(label.shape)
    # labels = np.asarray([dense_to_ont_hot(l, config.num_classes) for l in label])
    return DataSet(train_wave=train_wave, train_label=train_label, train_seqLength=train_seqLen,
                   valid_wave=valid_wave, valid_label=valid_label, valid_seqLength=valid_seqLen, valid_name=valid_name)


read_dataset()
# x = np.asarray([1, 2, 3, 4, 5])
# y = dense_to_ont_hot(x, 8)
# print(y)
