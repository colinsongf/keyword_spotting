# encoding: utf-8

"""

@author: ZiqiLiu


@file: reader.py

@time: 2017/5/11 下午12:33

@desc:
"""
import numpy as np
import tensorflow as tf
from utils.common import path_join
import os
from glob import glob
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import random_ops


def dense_to_ont_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, config, train_dir, valid_dir, mode='train'):
        self.config = config
        if mode == 'train':
            self.train_filename = glob(path_join(train_dir, '*.tfrecords'))
            print(path_join(train_dir, '*.tfrecords'))
            # self.train_filename = ['./data/mel_all/train/data00029.tfrecords',
            #                        './data/mel_all/train/data00030.tfrecords',
            #                        './data/mel_all/train/data00031.tfrecords']
            self.train_file_size = len(self.train_filename)
            if self.train_file_size == 0:
                raise Exception('train tfrecords not found')
            self.train_reader = tf.TFRecordReader(name='train_reader')
            self.epoch = self.epochs_completed
            print('file size', self.train_file_size)

        self.valid_filename = glob(path_join(valid_dir, '*.tfrecords'))
        self.valid_file_size = len(self.valid_filename)
        if self.valid_file_size == 0:
            raise Exception('valid tfrecords not found')
        self.valid_reader = tf.TFRecordReader(name='valid_reader')
        self.validation_size = len(self.valid_filename) * config.tfrecord_size
        print('validation size', self.validation_size)

    @property
    def epochs_completed(self):
        return self.train_reader.num_work_units_completed() // self.train_file_size

    @property
    def test(self):
        return self.train_reader.num_work_units_completed(), self.train_reader.num_records_produced()

    def train_filequeue_reader(self, filename_queue):
        (keys, values) = self.train_reader.read_up_to(filename_queue,
                                                      self.config.batch_size)
        context_features = {
            "seq_len": tf.FixedLenFeature([1], dtype=tf.int64),
        }
        audio_features = {
            "audio": tf.FixedLenSequenceFeature([self.config.num_features],
                                                dtype=tf.float32),
            self.config.label + '_label': tf.FixedLenSequenceFeature(
                [self.config.num_classes], dtype=tf.float32)
        }
        audio_list = []
        label_list = []
        len_list = []

        for i in range(self.config.batch_size):
            context, sequence = tf.parse_single_sequence_example(
                serialized=values[i],
                context_features=context_features,
                sequence_features=audio_features
            )
            audio = sequence['audio']
            label = sequence[self.config.label + '_label']
            seq_len = context['seq_len']
            audio_list.append(audio)
            label_list.append(label)
            len_list.append(seq_len)

        seq_lengths = tf.reshape(tf.stack(len_list, name='seq_lengths'),
                                 (-1,))

        return tf.stack(audio_list, name='input_audio'), tf.stack(label_list,
                                                                  name='input_label'), seq_lengths, keys

    def valid_filequeue_reader(self, filename_queue):
        (keys, values) = self.valid_reader.read_up_to(filename_queue,
                                                      self.config.batch_size)
        context_features = {
            "seq_len": tf.FixedLenFeature([1], dtype=tf.int64),
            "name": tf.FixedLenFeature([], dtype=tf.string),
            "correctness": tf.FixedLenFeature([1], dtype=tf.int64)
        }
        audio_features = {
            "audio": tf.FixedLenSequenceFeature([self.config.num_features],
                                                dtype=tf.float32)
        }
        audio_list = []
        len_list = []
        correct_list = []
        name_list = []

        for i in range(self.config.batch_size):
            context, sequence = tf.parse_single_sequence_example(
                serialized=values[i],
                context_features=context_features,
                sequence_features=audio_features
            )
            audio = sequence['audio']
            seq_len = context['seq_len']
            correct = context['correctness']
            name = context['name']
            audio_list.append(audio)
            len_list.append(seq_len)
            correct_list.append(correct)
            name_list.append(name)

        seq_lengths = tf.reshape(tf.stack(len_list), (-1,), name='seq_lengths')
        correctness = tf.reshape(tf.stack(correct_list), (-1,),
                                 name='correctness')
        name_tensor = tf.stack(name_list)

        return tf.stack(audio_list,
                        name='input_audio'), seq_lengths, correctness, name_tensor

    def string_input_queue(self, string_tensor, shuffle=True,
                           name=None, seed=None, capacity=16384):
        with ops.name_scope(name, "input_producer", [string_tensor]) as name:
            input_tensor = ops.convert_to_tensor(
                string_tensor, dtype=dtypes.string)
            if shuffle:
                input_tensor = random_ops.random_shuffle(input_tensor,
                                                         seed=seed)
            q = data_flow_ops.FIFOQueue(
                capacity=capacity,
                dtypes=[input_tensor.dtype.base_dtype])
            enq = tf.cond(tf.less(q.size(), 2),
                          lambda: q.enqueue_many([input_tensor]),
                          lambda: tf.no_op())
            return q, enq

    def batch_input_queue(self, shuffle=True):
        with tf.device('/cpu:0'):
            self.train_filename_queue, self.train_filequeue_enqueue_op = self.string_input_queue(
                self.train_filename, shuffle=shuffle, capacity=16384)

            audio, label, seq_len, keys = self.train_filequeue_reader(
                    self.train_filename_queue)

            stager = data_flow_ops.StagingArea(
                [tf.float32, tf.float32, tf.int64, tf.string],
                shapes=[(self.config.batch_size, None, self.config.num_features),
                        (self.config.batch_size, None, self.config.num_classes),
                        (self.config.batch_size), (None,)])

            stage_op = stager.put((audio, label, seq_len, keys))

            return stager, stage_op, self.train_filequeue_enqueue_op

    def valid_queue(self):
        with tf.device('/cpu:0'):
            self.valid_filename_queue, self.valid_filequeue_enqueue_op = self.string_input_queue(
                self.valid_filename, shuffle=False, capacity=16384)

            audio, seq_len, correctness, names = self.valid_filequeue_reader(
                self.valid_filename_queue)

            stager = data_flow_ops.StagingArea(
                [tf.float32, tf.int64, tf.int64, tf.string],
                shapes=[(self.config.batch_size, None, self.config.num_features),
                        (self.config.batch_size), (self.config.batch_size),
                        (None,)])

            stage_op = stager.put((audio, seq_len, correctness,names))

            return stager, stage_op, self.valid_filequeue_enqueue_op


def read_dataset(config, dtype=dtypes.float32):
    save_train_dir = config.train_path
    save_valid_dir = config.valid_path

    return DataSet(config=config, train_dir=save_train_dir,
                   valid_dir=save_valid_dir, mode=config.mode, )
