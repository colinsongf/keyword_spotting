# encoding: utf-8

"""

@author: ZiqiLiu


@file: reader.py

@time: 2017/5/11 下午12:33

@desc:
"""
import numpy as np
from tensorflow.python.framework import dtypes
import pickle
import os


def dense_to_ont_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, config, valid_wave, valid_labels, valid_seqLength, valid_name,
                 mode='train', train_wave=None, train_label=None, train_seqLength=None):
        self.config = config
        if mode == 'train':
            assert train_wave is not None
            self.wave = train_wave
            self.labels = train_label
            self.seqLength = train_seqLength

            self.train_size = len(self.wave)
            self.perm = np.arange(self.train_size)

        self.validation_size = len(valid_wave)
        print(self.validation_size)
        assert (self.validation_size % config.batch_size == 0)

        self.vi_wave = valid_wave
        self.vi_labels = valid_labels
        self.vi_seqLength = valid_seqLength
        self.valid_name = valid_name
        self._epochs_completed = 0
        self._index_in_epoch = 0

        self.valid_name_dict = {}
        for i, name in enumerate(valid_name):
            self.valid_name_dict[name] = i
        if len(valid_wave) < config.batch_size:
            self.vi_wave = self.padding(self.vi_wave, config.batch_size)
            self.vi_labels = self.padding(self.vi_labels, config.batch_size)
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

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def test_data(self, name=None):

        if name is not None:
            index = self.valid_name_dict[name]
            vi_wave = np.zeros(self.vi_wave.shape, dtype=np.float32)
            vi_label = np.zeros(self.vi_labels.shape, dtype=np.float32)
            vi_seq = np.zeros(self.vi_seqLength.shape, dtype=np.float32)
            vi_wave[0] = self.vi_wave[index]
            vi_label[0] = self.vi_labels[index]
            vi_seq[0] = self.vi_seqLength[index]
            return vi_wave, vi_label, vi_seq, self.valid_name
        else:
            perm = np.arange(self.config.batch_size)
            return self.vi_wave[perm], self.vi_labels[perm], \
                   self.vi_seqLength[perm], self.valid_name[:self.config.batch_size]

    def validate(self):
        for i in range(self.validation_size // self.config.batch_size):
            yield self.vi_wave[i * self.config.batch_size: (i + 1) * self.config.batch_size], \
                  self.vi_labels[i * self.config.batch_size: (i + 1) * self.config.batch_size], \
                  self.vi_seqLength[i * self.config.batch_size: (i + 1) * self.config.batch_size]

    def next_batch(self, shuffle=True):

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            np.random.shuffle(self.perm)
        # Go to the next epoch
        if start + self.config.batch_size > self.train_size:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num = self.train_size - start
            rest_part = self.perm[start:self.train_size]
            # Shuffle the data
            if shuffle:
                self.perm = np.arange(self.train_size)
                np.random.shuffle(self.perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = self.config.batch_size - rest_num
            end = self._index_in_epoch
            new_part = self.perm[start:end]
            temp_index = np.concatenate((rest_part, new_part), axis=0)
            return self.wave[temp_index], self.labels[temp_index], self.seqLength[temp_index]
        else:
            self._index_in_epoch += self.config.batch_size
            end = self._index_in_epoch
            return self.wave[self.perm[start:end]], self.labels[self.perm[start:end]], self.seqLength[
                self.perm[start:end]]


def read_dataset(config, dtype=dtypes.float32):
    data_dir = config.data_path
    save_train_dir = os.path.join(data_dir, 'train/')
    save_valid_dir = os.path.join(data_dir, 'valid/')

    train_wave = None
    train_label = None
    train_seqLen = None
    if config.mode == 'train':
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
    return DataSet(config=config, train_wave=train_wave, train_label=train_label, train_seqLength=train_seqLen,
                   mode=config.mode,
                   valid_wave=valid_wave, valid_labels=valid_label, valid_seqLength=valid_seqLen, valid_name=valid_name)


read_dataset(True)
# x = np.asarray([1, 2, 3, 4, 5])
# y = dense_to_ont_hot(x, 8)
# print(y)
