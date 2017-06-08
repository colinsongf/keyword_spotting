# encoding: utf-8

"""

@author: ZiqiLiu


@file: reader.py

@time: 2017/5/11 下午12:33

@desc:
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from utils.common import path_join
import pickle
import os
from glob import glob


def dense_to_ont_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, config, train_dir, valid_wave, valid_seqLength, valid_name, valid_correctness,
                 mode='train'):
        self.config = config
        if mode == 'train':
            filename = glob(path_join(train_dir, '*.tfrecords'))
            filename = sorted(filename)
            self.file_size = len(filename)
            if self.file_size == 0:
                raise Exception('tfrecords not found')
            print(filename)
            self.filename_queue = tf.train.string_input_producer(filename, config.max_epoch, shuffle=True,
                                                                 capacity=16384)
            self.reader = tf.TFRecordReader()

            self.train_size = len(filename) * config.tfrecord_size

            print('file size', self.file_size)

        self.validation_size = len(valid_wave)
        print(self.validation_size)
        # assert (self.validation_size % config.batch_size == 0)

        self.vi_wave = valid_wave
        self.vi_seqLength = valid_seqLength
        self.valid_name = valid_name
        self.valid_correctness = valid_correctness

        self.valid_name_dict = {}
        for i, name in enumerate(valid_name):
            self.valid_name_dict[name] = i

    @property
    def epochs_completed(self):
        return self.reader.num_records_produced() // self.train_size

    @property
    def test(self):
        return self.reader.num_work_units_completed(), self.reader.num_records_produced()

    def validate(self):
        for i in range(self.validation_size // self.config.batch_size):
            yield self.vi_wave[i * self.config.batch_size: (i + 1) * self.config.batch_size], \
                  self.vi_seqLength[i * self.config.batch_size: (i + 1) * self.config.batch_size], \
                  self.valid_correctness[i * self.config.batch_size: (i + 1) * self.config.batch_size], \
                  self.valid_name[i * self.config.batch_size: (i + 1) * self.config.batch_size]

    def next_batch(self, shuffle=True):
        (keys, values) = self.reader.read_up_to(self.filename_queue, self.config.batch_size)
        self.keys = keys
        context_features = {
            "seq_len": tf.FixedLenFeature([1], dtype=tf.int64),
        }
        audio_features = {
            "audio": tf.FixedLenSequenceFeature([self.config.num_features], dtype=tf.float32),
            "label": tf.FixedLenSequenceFeature([self.config.num_classes], dtype=tf.float32)
        }
        audio_list = []
        label_list = []
        self.len_list = []
        self.lab_len_list = []

        for i in range(self.config.batch_size):
            context, sequence = tf.parse_single_sequence_example(
                serialized=values[i],
                context_features=context_features,
                sequence_features=audio_features
            )
            audio = sequence['audio']
            label = sequence['label']
            # seq_len = context['seq_len'][0]
            seq_len = tf.shape(audio)[0]
            audio_list.append(audio)
            label_list.append(label)
            self.len_list.append(seq_len)
            self.lab_len_list.append(tf.shape(label)[0])
        seq_lengths = tf.stack(self.len_list, name='seq_lengths')
        self.max_length = tf.reduce_max(seq_lengths)
        self.max_la_len = tf.reduce_max(tf.stack(self.lab_len_list))

        new_audio_list = []
        self.new_label_list = []
        for i in range(self.config.batch_size):
            audio_padding = tf.zeros([1, self.config.num_features])
            audio_padding = tf.tile(audio_padding, [self.max_length - self.len_list[i], 1])
            label_padding = tf.zeros([1, self.config.num_classes])
            label_padding = tf.tile(label_padding, [self.max_la_len - self.lab_len_list[i], 1], name='fuck' + str(i))

            new_audio = tf.concat([audio_list[i], audio_padding], 0)
            new_audio_list.append(new_audio)
            new_label = tf.concat([label_list[i], label_padding], 0)
            self.new_label_list.append(new_label)
        # return max_length, len_list, keys
        return tf.stack(new_audio_list, name='input_audio'), tf.stack(self.new_label_list, name='input_label'), \
               seq_lengths, self.max_length, keys, self.len_list


def read_dataset(config, dtype=dtypes.float32):
    data_dir = config.data_path
    save_train_dir = path_join(data_dir, 'train/')
    save_valid_dir = path_join(data_dir, 'valid/')

    valid_wave = np.load(save_valid_dir + 'wave.npy')
    valid_seqLen = np.load(save_valid_dir + 'seqLen.npy')
    with open(save_valid_dir + 'filename.pkl', 'rb') as fs:
        valid_name = pickle.load(fs)
    with open(save_valid_dir + 'correctness.pkl', 'rb') as fs:
        valid_correctness = pickle.load(fs)
    # print(wave.shape)
    # print(label.shape)
    # labels = np.asarray([dense_to_ont_hot(l, config.num_classes) for l in label])
    return DataSet(config=config, train_dir=save_train_dir, mode=config.mode, valid_correctness=valid_correctness,
                   valid_wave=valid_wave, valid_seqLength=valid_seqLen, valid_name=valid_name)

# x = np.asarray([1, 2, 3, 4, 5])
# y = dense_to_ont_hot(x, 8)
# print(y)
