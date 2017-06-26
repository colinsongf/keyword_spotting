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
import pickle
import signal
import queue

import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.python.framework import graph_util

from config.config import get_config
from models.dynamic_rnn import DRNN, DeployModel
from reader import read_dataset
from utils.common import check_dir, path_join
from utils.prediction import predict, decode, moving_average, evaluate
from args import get_args

sys.dont_write_bytecode = True
DEBUG = False


def handler_stop_signals(signum, frame):
    global run
    run = False


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.golden = config.golden

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
                print(files)
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
            best_miss = 1
            best_false = 1
            accu_loss = 0
            accu_bg_loss = 0
            accu_key_loss = 0
            st_time = time.time()
            if os.path.exists(path_join(self.config.save_path, 'best.pkl')):
                with open(path_join(self.config.save_path, 'best.pkl'),
                          'rb') as f:
                    best_miss, best_false = pickle.load(f)
                    print('best miss', best_miss, 'best false', best_false)
            else:
                print('best not exist')

            check_dir(self.config.save_path)

            if self.config.mode == 'train':
                step = 0

                if self.config.reset_global:
                    sess.run(self.train_model.reset_global_step)

                def handler_stop_signals(signum, frame):
                    global run
                    run = False
                    if not DEBUG:
                        print(
                            'training shut down, total setp %s, the model will be save in %s' % (
                                step, self.config.save_path))
                        saver.save(sess, save_path=(
                            path_join(self.config.save_path, 'latest.ckpt')))
                        print('best miss rate:%f\tbest false rate %f' % (
                            best_miss, best_false))

                signal.signal(signal.SIGINT, handler_stop_signals)
                signal.signal(signal.SIGTERM, handler_stop_signals)

                best_list = []
                best_threshold = 0.22
                best_count = 0
                # (miss,false,step,best_count)

                last_time = time.time()
                all_va = [n.name for n in tf.global_variables()]
                for i in all_va:
                    print(i)
                try:
                    sess.run([self.train_model.stage_op,
                              self.train_model.input_filequeue_enqueue_op])

                    while self.epoch < self.config.max_epoch:

                        if not self.config.max_pooling_loss:
                            _, _, _, l, keys, lr, step = sess.run(
                                [self.train_model.train_op,
                                 self.train_model.stage_op,
                                 self.train_model.input_filequeue_enqueue_op,
                                 self.train_model.loss, self.train_model.keys,
                                 self.train_model.learning_rate,
                                 self.train_model.global_step])
                            epoch = sess.run([self.data.epoch])[0]

                        else:
                            _, _, _, l, xent_bg, xent_max, lr, step = sess.run(
                                [self.train_model.train_op,
                                 self.train_model.stage_op,
                                 self.train_model.input_filequeue_enqueue_op,
                                 self.train_model.loss,
                                 self.train_model.xent_background,
                                 self.train_model.xent_max_frame,
                                 self.train_model.learning_rate,
                                 self.train_model.global_step])
                            epoch = sess.run([self.data.epoch])[0]
                            accu_bg_loss += xent_bg
                            accu_key_loss += xent_max
                            # print(xent_bg, xent_max)

                        accu_loss += l
                        if epoch > self.epoch:
                            self.epoch += 1
                            print('accumulated loss', accu_loss)
                            accu_loss = 0
                            if config.max_pooling_loss:
                                print(accu_bg_loss, accu_key_loss)
                                accu_bg_loss = 0
                                accu_key_loss = 0
                        if step % config.valid_step == config.valid_step - 1:
                            print('epoch time ', (time.time() - last_time) / 60)
                            last_time = time.time()

                            miss_count = 0
                            false_count = 0
                            target_count = 0
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
                                    decode(p, self.config.word_interval,
                                           self.golden)
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
                            if config.max_pooling_loss:
                                print(xent_bg, xent_max)
                            print('learning rate:', lr, 'global step', step)
                            print('miss rate:' + str(miss_rate))
                            print('flase_accept_rate:' + str(false_accept_rate))
                            print(miss_count, '/', target_count)

                            if miss_rate + false_accept_rate < best_miss + best_false:
                                best_miss = miss_rate
                                best_false = false_accept_rate
                                saver.save(sess,
                                           save_path=(path_join(
                                               self.config.save_path,
                                               'best.ckpt')))
                                with open(path_join(
                                        self.config.save_path, 'best.pkl'),
                                        'wb') as f:
                                    best_tuple = (best_miss, best_false)
                                    pickle.dump(best_tuple, f)
                            if miss_rate + false_accept_rate < best_threshold:
                                best_count += 1
                                print('best_count', best_count)
                                best_list.append((miss_rate,
                                                  false_accept_rate, step,
                                                  best_count))
                                saver.save(sess,
                                           save_path=(path_join(
                                               self.config.save_path,
                                               'best' + str(
                                                   best_count) + '.ckpt')))

                    if not DEBUG:
                        print(
                            'training finished, total epoch %d, the model will be save in %s' % (
                                self.epoch, self.config.save_path))
                        saver.save(sess, save_path=(
                            path_join(self.config.save_path, 'latest.ckpt')))
                        print('best miss rate:%f\tbest false rate"%f' % (
                            best_miss, best_false))
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    pickle.dump(best_list, 'best_list.pkl')
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
                    # if i > 7:
                    #     break
                    ind = 14
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
                    result = [decode(p, self.config.word_interval, self.golden)
                              for p in
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
        config_path = path_join(self.config.graph_path, 'config.pkl')
        graph_path = path_join(self.config.graph_path, self.config.graph_name)
        import pickle
        pickle.dump(self.config, open(config_path, 'wb'))

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as session:
            with tf.variable_scope("model"):
                model = DeployModel(config=config)

            print('Graph build finished')
            # variable_names = [n.name for n in
            #                   tf.get_default_graph().as_graph_def().node]
            # for n in variable_names:
            #     print(n)

            saver = tf.train.Saver()
            saver.restore(session, save_path=path_join(self.config.model_path,
                                                       'latest.ckpt'))
            print("model restored from %s" % config.model_path)

            frozen_graph_def = graph_util.convert_variables_to_constants(
                session, session.graph.as_graph_def(),
                ['model/inputX', 'model/softmax', 'model/seqLength',
                 'model/fuck'])
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

    flags = get_args()
    for key in flags:
        if flags[key] is not None:
            if not hasattr(config, key):
                print("WARNING: Invalid override with attribute %s" % (key))
            else:
                setattr(config, key, flags[key])
    if not config.ktq:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    print(flags)
    runner = Runner(config)
    if config.mode == 'build':
        runner.build_graph()
    runner.run()
