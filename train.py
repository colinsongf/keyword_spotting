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

sys.dont_write_bytecode = True
import os
import numpy as np
import tensorflow as tf
from glob2 import glob
from models.dynamic_rnn import DRNN
from reader import read_dataset
import argparse
import time
from utils.common import check_dir, path_join

DEBUG = False


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.epoch = -1
        self.step = 0

        self.model = DRNN(self.config)
        self.model.config.show()
        self.data = read_dataset(self.config)

    def run(self):

        print('fuck')
        with tf.Session(graph=self.model.graph) as sess:
            # restore from stored models
            files = glob(path_join(self.config.model_path, '*.ckpt.*'))

            if len(files) > 0:
                self.model.saver.restore(sess, path_join(self.config.model_path, self.config.model_name))
                print(('Model restored from:' + self.config.model_path))
            else:
                print("Model doesn't exist.\nInitializing........")
                sess.run(self.model.initial_op)
            st_time = time.time()
            check_dir(self.config.working_path)
            if self.config.mode == 'train':
                try:
                    best_miss_rate = 1
                    while self.epoch < self.config.max_epoch:
                        self.step += 1
                        # if self.step > 1:
                        #     break
                        x, y, seqLengths = self.data.next_batch()

                        if not self.config.max_pooling_loss:
                            _, l = sess.run([self.model.optimizer, self.model.loss],
                                            feed_dict={self.model.inputX: x, self.model.inputY: y,
                                                       self.model.seqLengths: seqLengths})
                        else:
                            _, l, xent_bg, xent_max, max_log = sess.run(
                                [self.model.optimizer, self.model.loss, self.model.xent_background,
                                 self.model.xent_max_frame,
                                 self.model.masked_log_softmax],
                                feed_dict={self.model.inputX: x, self.model.inputY: y,
                                           self.model.seqLengths: seqLengths})
                            # _, max_frame, bk_label, lbs, mask_softmax = sess.run(
                            #     [self.model.optimizer, self.model.max_frame, self.model.background_label,
                            #      self.model.labels,
                            #      self.model.masked_log_softmax],
                            #     feed_dict={self.model.inputX: x, self.model.inputY: y,
                            #                self.model.seqLengths: seqLengths})
                            # print(mask_softmax.shape)
                            # print(max_frame.shape)
                            # np.set_printoptions(precision=4, threshold=np.inf, suppress=True)
                            # with open('logits.txt', 'w') as f:
                            #     f.write(str(max_frame[0]))
                            # with open('moving_avg.txt', 'w') as f:
                            #     f.write(str(bk_label[0]))
                            # with open('trigger.txt', 'w') as f:
                            #     f.write(str(lbs[0]))
                            # with open('label.txt', 'w') as f:
                            #     f.write(str(mask_softmax[0]))

                            # print(xent_bg, xent_max)
                            # np.set_printoptions(precision=4, threshold=np.inf, suppress=True)

                            # with open('logits.txt', 'w') as f:
                            #     f.write(str(max_log[0]))

                        if self.data.epochs_completed > self.epoch:
                            self.epoch += 1

                            miss_count = 0
                            false_count = 0
                            target_count = 0

                            for x, y, seqLengths, valid_correctness, names in self.data.validate():
                                logits, labels, seqLen = sess.run(
                                    [self.model.softmax, self.model.labels,
                                     self.model.seqLengths],
                                    feed_dict={self.model.inputX: x, self.model.inputY: y,
                                               self.model.seqLengths: seqLengths})

                                for i, logit in enumerate(logits):
                                    logit[seqLen[i]:] = 0

                                # print(len(logits), len(labels), len(seqLen))

                                moving_average = [
                                    self.moving_average(record, self.config.smoothing_window, padding=True)
                                    for record in logits]

                                # print(moving_average[0].shape)
                                prediction = [
                                    self.prediction(moving_avg, self.config.trigger_threshold, self.config.lockout)
                                    for moving_avg in moving_average]
                                # print(prediction[0].shape)

                                result = [self.decode(p, self.config.word_interval) for p in prediction]
                                miss, target, false_accept = self.correctness(result, valid_correctness)

                                miss_count += miss
                                target_count += target
                                false_count += false_accept
                                # print(miss_count, false_count)

                            miss_rate = miss_count / target_count
                            false_accept_rate = false_count / (self.data.validation_size - target_count)
                            print('--------------------------------')
                            print('epoch %d' % self.epoch)
                            print('loss:' + str(l))
                            print('miss rate:' + str(miss_rate))
                            print('flase_accept_rate:' + str(false_accept_rate))

                            if miss_rate + false_accept_rate < best_miss_rate:
                                best_miss_rate = miss_rate
                                self.model.saver.save(sess,
                                                      save_path=(path_join(self.config.working_path, 'best.ckpt')))

                except KeyboardInterrupt:
                    if not DEBUG:
                        print('training shut down, the model will be save in %s' % self.config.working_path)
                        self.model.saver.save(sess, save_path=(path_join(self.config.working_path, 'latest.ckpt')))
                        print('best miss rate:%f' % best_miss_rate)

                if not DEBUG:
                    print('training finished, total epoch %d, the model will be save in %s' % (
                        self.epoch, self.config.working_path))
                    self.model.saver.save(sess, save_path=(path_join(self.config.working_path, 'latest.ckpt')))
                    print('total time:%f hours' % ((time.time() - st_time) / 3600))
                    print('best miss rate:%f' % best_miss_rate)

            else:
                miss_count = 0
                false_count = 0
                target_count = 0
                total_count = 0

                iter = 0
                for x, y, seqLengths, valid_correctness, names in self.data.validate():
                    # print(names)
                    iter += 1
                    if iter != 1:
                        continue
                    ind = 6
                    np.set_printoptions(precision=4, threshold=np.inf, suppress=True)
                    print(str(names[ind]))
                    logits, labels, seqLen = sess.run(
                        [self.model.softmax, self.model.labels,
                         self.model.seqLengths],
                        feed_dict={self.model.inputX: x, self.model.inputY: y,
                                   self.model.seqLengths: seqLengths})
                    total_count += len(logits)
                    for i, logit in enumerate(logits):
                        logit[seqLen[i]:] = 0

                    # print(len(logits), len(labels), len(seqLen))
                    with open('logits.txt', 'w') as f:
                        f.write(str(logits[ind]))
                    with open('label.txt', 'w') as f:
                        f.write(str([labels[ind]]))
                    moving_average = [
                        self.moving_average(record, self.config.smoothing_window, padding=True)
                        for record in logits]

                    # print(moving_average[0].shape)
                    prediction = [
                        self.prediction(moving_avg, self.config.trigger_threshold, self.config.lockout)
                        for moving_avg in moving_average]
                    # print(prediction[0].shape)


                    with open('trigger.txt', 'w') as f:
                        f.write(str(prediction[ind]))
                    result = [self.decode(p, self.config.word_interval) for p in prediction]
                    miss, target, false_accept = self.correctness(result, valid_correctness)

                    miss_count += miss
                    target_count += target
                    false_count += false_accept

                    print(result[ind], valid_correctness[ind])
                    with open('moving_avg.txt', 'w') as f:
                        f.write(str(moving_average[ind]))

                # miss_rate = miss_count / target_count
                # false_accept_rate = false_count / total_count
                print('--------------------------------')
                print('miss rate: %d/%d' % (miss_count, target_count))
                print('flase_accept_rate: %d/%d' % (false_count, total_count))

    def prediction(self, moving_avg, threshold, lockout, f=None):
        if f is not None:
            print(f)
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
                j += 1
        return prediction

    def decode(self, prediction, word_interval):
        raw = [1]
        keyword = list(raw)
        # prediction based on moving_avg,shape(t,p),sth like one-hot, but can may overlapping
        # prediction = prediction[:, 1:]
        num_class = prediction.shape[1]
        len_frame = prediction.shape[0]
        pre = 0
        inter = 0

        target = keyword.pop()
        # try:
        for frame in prediction:
            if frame.sum() > 0:
                if pre == 0:
                    assert frame.sum() == 1
                    index = np.nonzero(frame)[0]
                    pre = index[0]
                    if index == target:
                        if inter < word_interval:
                            if len(keyword) == 0:
                                return 1
                            target = keyword.pop()
                            continue
                    keyword = list(raw)
                    target = keyword.pop()
                    inter = 0
                    if index == target:
                        target = keyword.pop()
                        continue
                else:
                    if frame[pre] == 1:
                        continue
                    else:
                        index = np.nonzero(frame)[0]
                        pre = index
                        if index == target:
                            if len(keyword) == 0:
                                return 1
                            target = keyword.pop()
                        else:
                            keyword = list(raw)
                            target = keyword.pop()
                            if index == target:
                                if len(keyword) == 0:
                                    return 1
                                target = keyword.pop()
                                continue
            else:
                if pre == 0:
                    if len(raw) - len(keyword) > 1:
                        inter += 1
                else:
                    pre = 0

        return 0
        # except Exception as e:
        #
        #     print('exception!!')
        #     np.set_printoptions(precision=4, threshold=np.inf, suppress=True)
        #     with open('test.txt', 'w') as f:
        #         f.write(str(prediction))
        #         return 1

    def correctness(self, result, target):
        assert len(result) == len(target)
        print(target)
        print(result)
        xor = [a ^ b for a, b in zip(target, result)]
        miss = sum([a & b for a, b in zip(xor, target)])
        false_accept = sum([a & b for a, b in zip(xor, result)])
        return miss, sum(target), false_accept

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
    config = get_config()
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='train: train model, ' +
                                       'valid: model validation, ',
                        default=None)
    parser.add_argument('-max', '--max_pooling_loss', help='1: maxpooling, ' +
                                                           '0: cross entropy,', type=int,
                        default=None)
    parser.add_argument('-m', '--model_path',
                        help='The  model path for restoring',
                        default=None)
    parser.add_argument('-w', '--working_path',
                        help='The  model path for  saving',
                        default=None)
    parser.add_argument('-g', '--gpu',
                        help='visable GPU',
                        default=None)
    parser.add_argument('-thres', '--threshold', help='threshold for trigger', type=float, default=None)
    parser.add_argument('--data_path', help='data path', default=None)
    parser.add_argument('--feature_num', help='data path', type=int, default=None)

    flags = parser.parse_args().__dict__

    for key in flags:
        if flags[key] is not None:
            if not hasattr(config, key):
                print("WARNING: Invalid override with attribute %s" % (key))
            else:
                setattr(config, key, flags[key])

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    print(flags)
    runner = Runner(config)
    runner.run()
