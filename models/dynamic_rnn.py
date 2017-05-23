# encoding: utf-8

'''

@author: ZiqiLiu


@file: dynamic_rnn.py

@time: 2017/5/18 上午11:04

@desc:
'''

# !/usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn import dynamic_rnn
import tensorflow.python.training.moving_averages

from utils.common import describe
from utils.common import dropout

cell_fn = core_rnn_cell_impl.LSTMCell


class DRNN(object):
    def __init__(self, config):
        self.config = config

        self.build_graph(config)

    @describe
    def build_graph(self, config):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(config.batch_size, None, config.num_features),
                                         name='inputX')  # [batchsize,len,features]
            self.inputY = tf.placeholder(tf.int32, shape=(config.batch_size, None), name='inputY')
            flatten_Y = tf.reshape(self.inputY, [-1])
            one_hot_Y = tf.one_hot(indices=flatten_Y, depth=config.num_classes, axis=1, dtype=dtypes.float32)
            self.labels = tf.reshape(one_hot_Y, (config.batch_size, -1, config.num_classes))
            # print(self.labels.shape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(config.batch_size), name='seqLengths')

            outputs = self.build_multi_dynamic_brnn(config, self.inputX, self.seqLengths)
            with tf.name_scope('fc-layer'):
                with tf.variable_scope('fc'):
                    weightsClasses = tf.Variable(
                        tf.truncated_normal([config.num_proj, config.num_classes], name='weightsClasses'))
                    biasesClasses = tf.Variable(tf.zeros([config.num_classes]), name='biasesClasses')
                    flatten_outputs = tf.reshape(outputs, (-1, config.num_proj))
                    flatten_logits = tf.matmul(flatten_outputs, weightsClasses) + biasesClasses
                    self.softmax = tf.reshape(tf.nn.softmax(flatten_logits),
                                              (config.batch_size, -1, config.num_classes))
                    flatten_labels = tf.reshape(self.labels, (-1, config.num_classes))
            # print(self.labels.shape)
            # print(self.logits.shape)

            self.xent_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=flatten_labels, logits=flatten_logits))

            # calculating maxpooling loss
            self.crop_softmax = tf.slice(self.softmax, [0, 0, 1], [-1, -1, -1])
            self.crop_labels = tf.slice(self.labels, [0, 0, 1], [-1, -1, -1])
            self.masked_softmax = self.crop_softmax * self.crop_labels
            self.one = tf.constant(1, dtype=tf.float32, shape=(config.batch_size,))  # shape (batchsize,)s
            self.max_frame = tf.reduce_max(self.masked_softmax, (1, 2))  # shape (batchsize,)
            self.xent_max_frame = -tf.reduce_sum(tf.log(self.max_frame + 1e-10) * self.one)

            self.background_softmax = tf.slice(self.softmax, [0, 0, 0], [-1, -1, 1])
            self.background_lable = tf.slice(self.labels, [0, 0, 0], [-1, -1, 1])
            self.xent_background = -tf.reduce_sum(tf.log(self.background_softmax) * self.background_lable)

            self.max_pooling_loss = self.xent_background + 100 * self.xent_max_frame

            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()

            if config.max_pooling_loss:
                self.loss = self.max_pooling_loss
            else:
                self.loss = self.xent_loss

            if config.grad_clip == -1:
                # not apply gradient clipping
                self.optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
            else:
                # apply gradient clipping
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op), config.grad_clip)
                opti = tf.train.AdamOptimizer(config.learning_rate)
                self.optimizer = opti.apply_gradients(zip(grads, self.var_trainable_op))
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def build_multi_dynamic_brnn(self,
                                 config,
                                 inputX,
                                 seqLengths, ):
        hid_input = inputX
        cell = cell_fn(num_units=config.hidden_size,
                       use_peepholes=True,
                       cell_clip=config.cell_clip,
                       initializer=None,
                       num_proj=config.num_proj, proj_clip=None,
                       forget_bias=1.0,
                       state_is_tuple=True,
                       activation=tf.tanh)
        for i in range(config.num_layers):
            outputs, output_states = dynamic_rnn(cell,
                                                 inputs=hid_input,
                                                 sequence_length=seqLengths,
                                                 initial_state=None,
                                                 dtype=tf.float32,
                                                 scope="drnn")

            # tensor of shape: [batch_size, max_time, input_size]
            hidden = outputs
            hidden = dropout(hidden, config.keep_prob, config.is_training)

            if i != config.num_layers - 1:
                hid_input = hidden

        return hidden


if __name__ == "__main__":
    pass
