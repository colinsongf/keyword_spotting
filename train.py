# encoding: utf-8

'''

@author: ZiqiLiu


@file: train.py

@time: 2017/5/18 上午11:03

@desc:
'''

# -*- coding:utf-8 -*-
# !/usr/bin/python

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from glob2 import glob
from tensorflow.python.framework import graph_util

from config.config import get_config
from models.dynamic_rnn import DRNN, DeployModel
from reader import read_dataset
from utils.common import check_dir, path_join
from utils.prediction import predict, decode, moving_average, evaluate

sys.dont_write_bytecode = True
DEBUG = False


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.step = 0

    def run(self):

        graph = tf.Graph()
        with graph.as_default(), tf.Session() as sess:
            print(tf.get_default_graph())
            self.data = read_dataset(self.config)

            if config.mode == 'train':
                with tf.variable_scope("model"):
                    self.train_model = DRNN(self.config,
                                            self.data.batch_input_queue(),
                                            is_train=True)
                    self.train_model.config.show()
                with tf.variable_scope("model", reuse=True):
                    self.valid_model = DRNN(self.config,
                                            self.data.valid_queue(),
                                            is_train=False)
            else:
                with tf.variable_scope("model", reuse=False):
                    self.valid_model = DRNN(self.config,
                                            self.data.valid_queue(),
                                            is_train=False)
            saver = tf.train.Saver()

            # restore from stored models
            files = glob(path_join(self.config.model_path, '*.ckpt.*'))

            if len(files) > 0:
                saver.restore(sess, path_join(self.config.model_path,
                                              self.config.model_name))
                print(('Model restored from:' + self.config.model_path))
            else:
                print("Model doesn't exist.\nInitializing........")
                sess.run(tf.global_variables_initializer())

            sess.run(tf.local_variables_initializer())
            tf.Graph.finalize(graph)
            variable_names = [n.name for n in
                              tf.get_default_graph().as_graph_def().node]
            # glo=sess.run([tf.global_variables()])
            # for n in variable_names:
            #     print(n)

            st_time = time.time()
            check_dir(self.config.model_path)
            check_dir(self.config.working_path)

            if self.config.mode == 'train':
                best_miss = 1
                best_false = 1
                last_time = time.time()
                sess.run([self.valid_model.stage_op,
                          self.valid_model.input_filequeue_enqueue_op])
                try:
                    sess.run([self.train_model.stage_op,
                              self.train_model.input_filequeue_enqueue_op])

                    while self.epoch < self.config.max_epoch:
                        self.step += 1
                        # if self.step > 1:
                        #     break
                        if not self.config.max_pooling_loss:
                            _, _, _, l, keys, labels = sess.run(
                                [self.train_model.optimizer,
                                 self.train_model.stage_op,
                                 self.train_model.input_filequeue_enqueue_op,
                                 self.train_model.loss, self.train_model.keys,
                                 self.train_model.labels])
                            epoch = sess.run([self.data.epoch])[0]
                            # np.set_printoptions(precision=4, threshold=np.inf,
                            #                     suppress=True)
                            # for la in labels:
                            #     if la[:, 2].sum()>0:
                            #         print('fuck')
                            # print(la[:, 2])
                            # print(keys[0])
                        else:
                            _, _, _, l, xent_bg, xent_max = sess.run(
                                [self.train_model.optimizer,
                                 self.train_model.stage_op,
                                 self.train_model.input_filequeue_enqueue_op,
                                 self.train_model.loss,
                                 self.train_model.xent_background,
                                 self.train_model.xent_max_frame])
                            epoch = sess.run([self.data.epoch])[0]

                        # if epoch > self.epoch:
                        #     print('epoch', self.epoch)
                        #     self.epoch += 1
                        #     continue
                        if epoch > self.epoch:
                            print('epoch time ', (time.time() - last_time) / 60)
                            last_time = time.time()
                            self.epoch += 1

                            miss_count = 0
                            false_count = 0
                            target_count = 0
                            ind = 3
                            for i in range(self.data.valid_file_size):
                                logits, seqLen, correctness, names, _, _ = sess.run(
                                    [self.valid_model.softmax,
                                     self.valid_model.seqLengths,
                                     self.valid_model.correctness,
                                     self.valid_model.names,
                                     self.valid_model.stage_op,
                                     self.valid_model.input_filequeue_enqueue_op])
                                for j, logit in enumerate(logits):
                                    logit[seqLen[j]:] = 0

                                # print(len(logits), len(labels), len(seqLen))
                                with open('logits.txt', 'w') as f:
                                    f.write(str(logits[ind]))
                                moving_avg = [moving_average(record,
                                                             self.config.smoothing_window,
                                                             padding=True)
                                              for record in logits]

                                prediction = [
                                    predict(avg, self.config.trigger_threshold,
                                            self.config.lockout)
                                    for avg in moving_avg]
                                # print(prediction[0].shape)

                                result = [
                                    decode(p, self.config.word_interval)
                                    for p in prediction]
                                miss, target, false_accept = evaluate(
                                    result, correctness.tolist())

                                miss_count += miss
                                target_count += target
                                false_count += false_accept
                                # print(miss_count, false_count)

                            miss_rate = miss_count / target_count
                            false_accept_rate = false_count / (
                                self.data.validation_size - target_count)
                            print('--------------------------------')
                            print('epoch %d' % self.epoch)
                            print('loss:' + str(l))
                            print('miss rate:' + str(miss_rate))
                            print('flase_accept_rate:' + str(false_accept_rate))
                            print(miss_count, '/', target_count)

                            if miss_rate + false_accept_rate < best_miss + best_false:
                                best_miss = miss_rate
                                best_false = false_accept_rate
                                saver.save(sess,
                                           save_path=(path_join(
                                               self.config.working_path,
                                               'best.ckpt')))
                    if not DEBUG:
                        print(
                            'training finished, total epoch %d, the model will be save in %s' % (
                                self.epoch, self.config.model_paths))
                        saver.save(sess, save_path=(
                            path_join(self.config.model_path, 'latest.ckpt')))
                        print('best miss rate:%f\tbest false rate"%f' % (
                            best_miss, best_false))
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                except KeyboardInterrupt:
                    if not DEBUG:
                        print(
                            'training shut down, total setp %s, the model will be save in %s' % (
                                self.step, self.config.model_path))
                        saver.save(sess, save_path=(
                            path_join(self.config.model_path, 'latest.ckpt')))
                        print('best miss rate:%f\tbest false rate %f' % (
                            best_miss, best_false))
                finally:
                    print('total time:%f hours' % (
                        (time.time() - st_time) / 3600))
                    # When done, ask the threads to stop.

            else:
                miss_count = 0
                false_count = 0
                target_count = 0
                total_count = 0

                iter = 0
                for i in range(self.data.valid_file_size):
                    # if i > 3:
                    #     break
                    ind = 3
                    logits, seqLen, correctness, names, _, _ = sess.run(
                        [self.valid_model.softmax,
                         self.valid_model.seqLengths,
                         self.valid_model.correctness,
                         self.valid_model.names,
                         self.valid_model.stage_op,
                         self.valid_model.input_filequeue_enqueue_op])

                    np.set_printoptions(precision=4, threshold=np.inf,
                                        suppress=True)
                    print(names[ind].decode('utf-8'))
                    total_count += len(logits)
                    for j, logit in enumerate(logits):
                        logit[seqLen[j]:] = 0

                    # print(len(logits), len(labels), len(seqLen))
                    with open('logits.txt', 'w') as f:
                        f.write(str(logits[ind]))
                    moving_avg = [
                        moving_average(record,
                                       self.config.smoothing_window,
                                       padding=True)
                        for record in logits]
                    prediction = [
                        predict(avg, self.config.trigger_threshold,
                                self.config.lockout)
                        for avg in moving_avg]

                    with open('trigger.txt', 'w') as f:
                        f.write(str(prediction[ind]))
                    result = [decode(p, self.config.word_interval) for p in
                              prediction]
                    miss, target, false_accept = evaluate(result,
                                                          correctness.tolist())

                    miss_count += miss
                    target_count += target
                    false_count += false_accept

                    print(result[ind], correctness[ind])
                    with open('moving_avg.txt', 'w') as f:
                        f.write(str(moving_avg[ind]))

                # miss_rate = miss_count / target_count
                # false_accept_rate = false_count / total_count
                print('--------------------------------')
                print('miss rate: %d/%d' % (miss_count, target_count))
                print('flase_accept_rate: %d/%d' % (
                    false_count, total_count - target_count))

    def build_graph(self):
        config_path = path_join(self.config.working_path, 'config.pkl')
        graph_path = path_join(self.config.working_path, 'graph.pb')
        import pickle
        pickle.dump(self.config, open(config_path, 'wb'))

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as session:
            with tf.variable_scope("model"):
                model = DeployModel(config=config)

            print('Graph build finished')

            saver = tf.train.Saver()
            saver.restore(session, save_path=path_join(self.config.model_path,
                                                       'latest.ckpt'))
            print("model restored from %s" % (config.model_path))

            frozen_graph_def = graph_util.convert_variables_to_constants(
                session, session.graph.as_graph_def(),
                ['model/softmax', 'model/seqLength'])
            tf.train.write_graph(
                frozen_graph_def,
                os.path.dirname(graph_path),
                os.path.basename(graph_path),
                as_text=False,
            )
            try:
                tf.import_graph_def(frozen_graph_def, name="")
            except Exception as e:
                print("!!!!Import graph meet error: ", e)
                exit()
            print('graph saved in %s' % graph_path)


if __name__ == '__main__':
    config = get_config()
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='train: train model, ' +
                                       'valid: model validation, ',
                        default=None)
    parser.add_argument('-max', '--max_pooling_loss', help='1: maxpooling, ' +
                                                           '0: cross entropy,',
                        type=int,
                        default=None)
    parser.add_argument('-m', '--model_path',
                        help='The  model path for restoring',
                        default=None)
    parser.add_argument('-w', '--working_path',
                        help='The  model path for  saving',
                        default=None)
    parser.add_argument('-g', '--gpu',
                        help='visible GPU',
                        default=None)
    parser.add_argument('-thres', '--threshold', help='threshold for trigger',
                        type=float, default=None)
    parser.add_argument('--data_path', help='data path', default=None)
    parser.add_argument('--feature_num', help='data path', type=int,
                        default=None)

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
    # runner.run()
    runner.build_graph()
