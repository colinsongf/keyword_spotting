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
from tensorflow.python.ops.rnn import dynamic_rnn

from utils.common import describe

cell_fn = core_rnn_cell_impl.LSTMCell


# noinspection PyAttributeOutsideInit,SpellCheckingInspection
class DRNN(object):
    def __init__(self, config, input, is_train):
        self.config = config
        if is_train:
            stager, self.stage_op, self.input_filequeue_enqueue_op = input
            # we only use 1 gpu
            self.inputX, self.labels, self.seqLengths, self.keys = stager.get()
            self.build_graph(config, is_train)
        else:
            stager, self.stage_op, self.input_filequeue_enqueue_op = input
            self.inputX, self.seqLengths, self.correctness, self.names = stager.get()
            self.build_graph(config, is_train)

    @describe
    def build_graph(self, config, is_train):

        outputs = build_multi_dynamic_rnn(config, self.inputX,
                                          self.seqLengths)
        with tf.name_scope('fc-layer'):
            if config.use_project:
                weightsClasses = tf.get_variable(name='weightsClasses',
                                                 initializer=tf.truncated_normal(
                                                     [config.num_proj,
                                                      config.num_classes]))
                flatten_outputs = tf.reshape(outputs, (-1, config.num_proj))
            else:
                weightsClasses = tf.get_variable(name='weightsClasses',
                                                 initializer=tf.truncated_normal(
                                                     [config.hidden_size,
                                                      config.num_classes]))
                flatten_outputs = tf.reshape(outputs,
                                             (-1, config.hidden_size))
            biasesClasses = tf.get_variable(name='biasesClasses',
                                            initializer=tf.truncated_normal(
                                                [config.num_classes]))

        flatten_logits = tf.matmul(flatten_outputs,
                                   weightsClasses) + biasesClasses
        self.softmax = tf.reshape(tf.nn.softmax(flatten_logits),
                                  (config.batch_size, -1,
                                   config.num_classes))
        if is_train:
            flatten_labels = tf.reshape(self.labels,
                                        (-1, config.num_classes))
            # self.xent_loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(labels=flatten_labels,
            #                                             logits=flatten_logits))
            self.xent_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(labels=flatten_labels,
                                                        logits=flatten_logits))
            # calculating maxpooling loss
            self.log_softmax = -tf.log(self.softmax)
            self.crop_log_softmax = tf.slice(self.log_softmax, [0, 0, 1],
                                             [-1, -1, -1])
            self.crop_labels = tf.slice(self.labels, [0, 0, 1], [-1, -1, -1])
            self.masked_log_softmax = self.crop_log_softmax * self.crop_labels
            self.segment_len = tf.count_nonzero(self.masked_log_softmax, 1,
                                                dtype=tf.float32)  # shape (batchsize,class_num)
            self.segment_len_sum = tf.reduce_sum(self.segment_len, axis=1)
            self.max_frame = tf.reduce_max(self.masked_log_softmax,
                                           1)  # shape (batchsize,class_num)
            self.xent_max_frame = tf.reduce_sum(self.max_frame)
            self.background_log_softmax = tf.slice(self.log_softmax, [0, 0, 0],
                                                   [-1, -1, 1])
            self.background_label = tf.slice(self.labels, [0, 0, 0],
                                             [-1, -1, 1])
            if config.max_pooling_standardize:
                self.xent_background = tf.reduce_sum(
                    tf.reduce_sum(
                        self.background_log_softmax * self.background_label,
                        (1, 2)) / (tf.cast(self.seqLengths,
                                           tf.float32) - self.segment_len_sum))
            else:
                self.xent_background = tf.reduce_sum(
                    self.background_log_softmax * self.background_label)

            self.flatten_masked_softmax = tf.reshape(self.masked_log_softmax,
                                                     (config.batch_size, -1))
            self.max_index = tf.arg_max(self.flatten_masked_softmax, 1)

            self.max_pooling_loss = self.xent_background + self.xent_max_frame

            self.global_step = tf.Variable(0, trainable=False)
            self.reset_global_step = tf.assign(self.global_step, 1)
            self.learning_rate = tf.train.exponential_decay(
                config.learning_rate, self.global_step, self.config.decay_step,
                self.config.lr_decay, name='lr')

            if config.max_pooling_loss:
                self.loss = self.max_pooling_loss
            else:
                self.loss = self.xent_loss

            if self.config.optimizer == 'sgd':
                self.train_op = tf.train.GradientDescentOptimizer(
                    self.learning_rate)
            elif self.config.optimizer == 'adam':
                self.train_op = tf.train.AdamOptimizer(self.learning_rate)
            elif self.config.optimizer == 'nesterov':
                self.train_op = tf.train.MomentumOptimizer(
                    self.learning_rate, 0.9, use_nesterov=True)
            else:
                raise Exception("optimizer not found")

            if config.grad_clip == -1:
                # not apply gradient clipping
                self.train_op = self.train_op.minimize(self.loss,
                                                       self.global_step)
            else:
                # apply gradient clipping
                self.var_trainable_op = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(
                    tf.gradients(self.loss, self.var_trainable_op),
                    config.grad_clip)
                self.train_op = self.train_op.apply_gradients(
                    zip(grads, self.var_trainable_op),
                    global_step=self.global_step)


class DeployModel(object):
    def __init__(self, config):
        """
        In deployment, we use placeholder to get input. Only inference part
        are built. seq_lengths, rnn_states, rnn_outputs, ctc_decode_inputs
        are exposed for streaming decoding. All operators are placed in CPU.
        Padding should be done before data is fed.
        """

        # input place holder
        with tf.device('/cpu:0'):
            self.inputX = tf.placeholder(dtype=tf.float32,
                                         shape=[None, config.num_features],
                                         name='inputX')
            inputX = tf.expand_dims(self.inputX, 0, name='reshape_inputX')
            self.fuck = tf.identity(inputX, name='fuck')

            self.seqLength = tf.placeholder(dtype=tf.int32, shape=[1],
                                            name='seqLength')

            rnn_outputs = build_multi_dynamic_rnn(config, inputX,
                                                  self.seqLength)
            with tf.name_scope('fc-layer'):
                if config.use_project:
                    weightsClasses = tf.get_variable(name='weightsClasses',
                                                     initializer=tf.truncated_normal(
                                                         [config.num_proj,
                                                          config.num_classes]))
                    flatten_outputs = tf.reshape(rnn_outputs,
                                                 (-1, config.num_proj))
                else:
                    weightsClasses = tf.get_variable(name='weightsClasses',
                                                     initializer=tf.truncated_normal(
                                                         [config.hidden_size,
                                                          config.num_classes]))
                    flatten_outputs = tf.reshape(rnn_outputs,
                                                 (-1, config.hidden_size))
                biasesClasses = tf.get_variable(name='biasesClasses',
                                                initializer=tf.truncated_normal(
                                                    [config.num_classes]))

            flatten_logits = tf.matmul(flatten_outputs,
                                       weightsClasses) + biasesClasses
            self.softmax = tf.nn.softmax(flatten_logits, name='softmax')


def build_multi_dynamic_rnn(config,
                            inputX,
                            seqLengths):
    hid_input = inputX
    print(tf.get_variable_scope().reuse)
    cell = cell_fn(num_units=config.hidden_size,
                   use_peepholes=True,
                   cell_clip=config.cell_clip,
                   initializer=tf.contrib.layers.xavier_initializer(),
                   num_proj=config.num_proj if config.use_project else None,
                   proj_clip=None,
                   forget_bias=1.0,
                   state_is_tuple=True,
                   activation=tf.tanh,
                   reuse=tf.get_variable_scope().reuse
                   )
    for i in range(config.num_layers):
        outputs, output_states = dynamic_rnn(cell,
                                             inputs=hid_input,
                                             sequence_length=seqLengths,
                                             initial_state=None,
                                             dtype=tf.float32,
                                             scope="drnn")

        # tensor of shape: [batch_size, max_time, input_size]
        hidden = outputs
        if config.mode == 'train':
            hidden = dropout(hidden, config.keep_prob)

        if i != config.num_layers - 1:
            hid_input = hidden

    return hidden


def dropout(x, keep_prob):
    """ Apply dropout to a tensor
    """
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob,
                                     is_training=True)


if __name__ == "__main__":
    pass
