# encoding: utf-8

'''

@author: ZiqiLiu


@file: train.py

@time: 2017/5/18 上午11:03

@desc:
'''

# -*- coding:utf-8 -*-
# !/usr/bin/python

import sys
from config.config import get_config
from data_test import read_dataset

sys.dont_write_bytecode = True

import time
import datetime
import os

import numpy as np
import tensorflow as tf
from glob2 import glob
from models.dynamic_rnn import DRNN
from functools import reduce
from data_test import read_dataset
import time

model_path = './params/latest.ckpt'
# model_path = None
save_path = './params/'
DEBUG = False


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Runner(object):
    def __init__(self):
        self.config = get_config()

    # @describe
    # def load_data(self, args, mode, type):
    #     if mode == 'train':
    #         return load_batched_data(train_mfcc_dir, train_label_dir, batch_size, mode, type)
    #     elif mode == 'test':
    #         return load_batched_data(test_mfcc_dir, test_label_dir, batch_size, mode, type)
    #     else:
    #         raise TypeError('mode should be train or test.')

    def run(self):
        # load data
        model = DRNN(self.config)
        global model_path
        model.config.show()
        self.data = read_dataset()

        print('fuck')
        with tf.Session(graph=model.graph) as sess:
            # restore from stored models
            if model_path is not None:
                model.saver.restore(sess, model_path)
                print(('Model restored from:' + model_path))
            else:
                files = glob(save_path + '*')
                if len(files) > 0:
                    print(files)
                    raise Exception(
                        'existing param files. Please move them before initializing, otherwise will overwrite')
                print('Initializing')
                sess.run(model.initial_op)
            st_time = time.time()
            if self.config.is_training:
                try:
                    for step in range(10001):
                        x, y, seqLengths = self.data.next_batch(self.config.batch_size, False)
                        if not self.config.max_pooling_loss:
                            _, l = sess.run([model.optimizer, model.loss], feed_dict={model.inputX: x, model.inputY: y,
                                                                                      model.seqLengths: seqLengths})
                        else:
                            _, l, xent_bg, xent_max, max_index = sess.run(
                                [model.optimizer, model.loss, model.xent_background, model.xent_max_frame,
                                 model.max_index],
                                feed_dict={model.inputX: x, model.inputY: y,
                                           model.seqLengths: seqLengths})
                            print('step', step, xent_bg, xent_max)
                            print(max_index)

                        if step % 200 == 0:
                            x, y, seqLengths, name = self.data.test_data()
                            _, logits, labels = sess.run([model.optimizer, model.softmax, model.labels],
                                                         feed_dict={model.inputX: x, model.inputY: y,
                                                                    model.seqLengths: seqLengths})
                            logits, labels = map((lambda a: a[:8]), (logits, labels))
                            print(len(logits))
                            print(len(labels))

                            moving_average = [self.moving_average(record, self.config.smoothing_window, padding=True)
                                              for record in logits]

                            # print(moving_average[0].shape)
                            prediction = [
                                self.prediction(moving_avg, self.config.trigger_threshold, self.config.lockout)
                                for moving_avg in moving_average]
                            # print(prediction[0].shape)

                            assert len(prediction) == len(labels)
                            correctness = [self.correctness(pred, label, self.config.latency) for pred, label in
                                           zip(prediction, labels)]
                            correctness = reduce(lambda a, b: a + b, np.asarray(correctness))
                            miss_rate, false_accept_rate = correctness / self.config.validation_size
                            print('--------------------------------')
                            print('step %d' % step)
                            print('loss:' + str(l))
                            print('miss rate:' + str(miss_rate))
                            print('flase_accept_rate:' + str(false_accept_rate))
                except KeyboardInterrupt:
                    if not DEBUG:
                        print('training shut down, the model will be save in %s' % save_path)
                        model.saver.save(sess, save_path=(save_path + 'latest.ckpt'))

                print('training finished, total step %d, the model will be save in %s' % (step, save_path))
                model.saver.save(sess, save_path=(save_path + 'latest.ckpt'))
            else:

                x, y, seqLengths, names = self.data.test_data()

                # x, y, seqLengths, names = self.data.test_data(self.config.validation_size,                                                              's_F193089BC92BAFDF_你好你是傻逼吗.wav')

                # print(len(seqLengths))

                _, logits, labels = sess.run([model.optimizer, model.softmax, model.labels],
                                             feed_dict={model.inputX: x, model.inputY: y,
                                                        model.seqLengths: seqLengths})
                logits, labels = map((lambda a: a[:8]), (logits, labels))

                moving_average = [self.moving_average(record, self.config.smoothing_window, padding=True)
                                  for record in logits]

                # print(len(moving_average))

                prediction = [
                    self.prediction(moving_avg, self.config.trigger_threshold, self.config.lockout)
                    for moving_avg in moving_average]
                # print(prediction[0].shape)


                ind = 0
                np.set_printoptions(precision=4, threshold=np.inf, suppress=True)
                print(str(names[ind]))

                with open('logits.txt', 'w') as f:
                    f.write(str(logits[ind]))
                with open('moving_avg.txt', 'w') as f:
                    f.write(str(moving_average[ind]))
                with open('trigger.txt', 'w') as f:
                    f.write(str(prediction[ind]))
                with open('label.txt', 'w') as f:
                    f.write(str([labels[ind]]))

                assert len(prediction) == len(labels)
                correctness = [self.correctness(pred, label, self.config.latency) for pred, label in
                               zip(prediction, labels)]
                correctness = reduce(lambda a, b: a + b, np.asarray(correctness))
                miss_rate, false_accept_rate = correctness / self.config.validation_size
                print('miss rate:' + str(miss_rate))
                print('flase_accept_rate:' + str(false_accept_rate))

    def prediction(self, moving_avg, threshold, lockout):
        # array is 2D array, moving avg for one record, whose shape is (t,p)
        # label is one-hot, which is also the same size of moving_avg
        # we assume label[0] is background
        # return a trigger array, the same size (t,p) (it's sth like mask)
        # print('=' * 50)
        num_class = moving_avg.shape[1]
        len_frame = moving_avg.shape[0]
        # print(num_class, len_frame)
        prediction = np.zeros(moving_avg.shape, dtype=np.float32)
        for i in range(1, num_class):
            j = 0
            while j < len_frame:
                if moving_avg[j][i] > threshold:
                    prediction[j][i] = 1
                    # print('lockout')
                    j += lockout
                else:
                    j += 1
        return prediction

    def correctness(self, prediction, label, latency):
        # for one record

        label[latency:, :] = label[latency:, :] + label[:-latency, :]
        label = np.clip(label, 0, 1)
        correct_trigger = prediction * label
        label_correct = 0
        # this only work for wav that keyword only appear once
        for i in range(1, label.shape[1]):
            if label[:, i].sum() > 0:
                label_correct += 1

        correct_trigger_num = correct_trigger.sum()

        false_accept = prediction.sum() - correct_trigger_num
        miss = label_correct - correct_trigger_num

        print('%d\t%d\t%d' % (label_correct, miss, false_accept))

        return np.asanyarray([miss, false_accept])

    def moving_average(self, array, n=5, padding=True):
        # array is 2D array, logits for one record, shape (t,p)
        # return shape (t,p)
        if n % 2 == 0:
            raise Exception('n must be odd')
        if len(array.shape) != 2:
            raise Exception('must be 2-D array.')
        if n > array.shape[0]:
            raise Exception('n larger than array length. the shape:' + str(array.shape))
        if padding:
            pad_num = n // 2
            array = np.pad(array=array, pad_width=((pad_num, pad_num), (0, 0)), mode='constant', constant_values=0)
        array = np.asarray([np.sum(array[i:i + n, :], axis=0) for i in range(len(array) - 2 * pad_num)]) / n
        return array


if __name__ == '__main__':
    runner = Runner()
    runner.run()
