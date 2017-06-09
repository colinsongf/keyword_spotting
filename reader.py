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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


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
            for f in filename:
                print(f)
            self.filename = sorted(filename)
            self.file_size = len(filename)
            if self.file_size == 0:
                raise Exception('tfrecords not found')
            print(filename)

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

    def filequeue_reader(self, filename_queue):
        (keys, values) = self.reader.read_up_to(filename_queue, self.config.batch_size)
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

        # return max_length, len_list, keys
        return tf.stack(audio_list, name='input_audio'), tf.stack(label_list, name='input_label'), \
               seq_lengths, keys

    def string_input_queue(self, string_tensor, shuffle=True,
                           name=None, seed=None, capacity=16384):
        with ops.name_scope(name, "input_producer", [string_tensor]) as name:
            input_tensor = ops.convert_to_tensor(
                string_tensor, dtype=dtypes.string)
            if shuffle:
                input_tensor = random_ops.random_shuffle(input_tensor, seed=seed)
            q = data_flow_ops.FIFOQueue(
                capacity=capacity,
                dtypes=[input_tensor.dtype.base_dtype])
            enq = tf.cond(tf.less(q.size(), 2),
                          lambda: q.enqueue_many([input_tensor]),
                          lambda: tf.no_op())
            return q, enq

    def batch_input_queue(self, shuffle):
        self.filename_queue, self.data_filequeue_enqueue_op = self.string_input_queue(self.filename, shuffle=True,
                                                                                      capacity=16384)
        with tf.device('/cpu:0'):
            audio, label, seq_len, keys = self.filequeue_reader(self.filename_queue)

        stagers = []
        stage_ops = []

        stager = data_flow_ops.StagingArea(
            [tf.int64, tf.int64, tf.int64, tf.float32, tf.int32])
        stage = stager.put(
            [audio, label, seq_len, keys])
        stagers.append(stager)
        stage_ops.append(stage)

        stage_op = tf.group(*stage_ops)

        return stagers, stage_op, self.data_filequeue_enqueue_op


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

