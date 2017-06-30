# encoding: utf-8

'''

@author: ZiqiLiu


@file: dynamic_rnn.py

@time: 2017/5/18 上午11:04

@desc:
'''

# !/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
import math

from utils.common import describe
from positional_encoding import positional_encoding_op


def self_attention(inputs, config, scope_name='self_attention'):
    # inputs - [batch_size, time_steps, model_size]
    # return - [batch_size, time_steps, model_size]
    assert (config.model_size % config.multi_head_num == 0)
    head_size = config.model_size // config.multi_head_num
    outputs = []
    with tf.variable_scope(scope_name):
        combined = tf.layers.conv2d(
            tf.expand_dims(inputs, 2), 3 * config.model_size,
            (1, 1), name="qkv_transform")
        q, k, v = tf.split(
            tf.squeeze(combined, 2),
            [config.model_size, config.model_size, config.model_size],
            axis=2)

        q_ = tf.concat(tf.split(q, config.multi_head_num, axis=2),
                       axis=0)  # (h * B, T, N / h)
        k_ = tf.concat(tf.split(k, config.multi_head_num, axis=2),
                       axis=0)  # (h * B, T, N / h)
        v_ = tf.concat(tf.split(v, config.multi_head_num, axis=2),
                       axis=0)  # (h * B, T, N / h)

        a = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # (h * B, T, T)
        a = tf.nn.softmax(a / math.sqrt(head_size))
        a = tf.nn.dropout(a, config.keep_prob)
        a = tf.matmul(a, v_)  # [h * B, T, N / h]

        outputs = tf.concat(tf.split(a, config.multi_head_num, axis=0),
                            axis=2)  # [B, T, N]

    return outputs


def feed_forward(inputs, config, scope_name='feed_forward'):
    # inputs - [batch_size, time_steps, model_size]
    # return - [batch_size, time_steps, model_size]
    with tf.variable_scope(scope_name):
        inners = tf.layers.conv2d(
            tf.expand_dims(inputs, 2), config.feed_forward_inner_size,
            (1, 1), activation=tf.nn.relu, name="conv1")  # [B, T, 1, F]
        outputs = tf.layers.conv2d(
            inners, config.model_size, (1, 1), name="conv2")  # [B, T, 1, F]
    return tf.squeeze(outputs, 2)


def inference(inputs, seqLengths, config):
    # positional encoding
    max_length = tf.shape(inputs)[1]
    inputs = tf.layers.conv2d(
        tf.expand_dims(inputs, 2), config.model_size, (1, 1),
        name='input_linear_trans')  # [B, T, 1, F]
    inputs = tf.squeeze(inputs, 2)  # [B, T, F]

    pe = positional_encoding_op.positional_encoding(
        max_length, config.model_size)
    inputs = inputs + pe

    layer_inputs = inputs
    for j in range(config.num_layers):
        with tf.variable_scope('layer_%d' % j):
            # self attention sub-layer
            attention_outputs = self_attention(layer_inputs, config)
            attention_outputs = tf.nn.dropout(
                attention_outputs, config.keep_prob)
            # add and norm
            feed_forward_inputs = tf.contrib.layers.layer_norm(
                attention_outputs + layer_inputs)
            # feed forward sub-layer
            feed_forward_outputs = feed_forward(feed_forward_inputs, config)
            feed_forward_outputs = tf.nn.dropout(
                feed_forward_outputs, config.keep_prob)
            # add and norm
            layer_outputs = tf.contrib.layers.layer_norm(
                feed_forward_outputs + feed_forward_inputs)
            layer_inputs = layer_outputs

    outputs = tf.layers.conv2d(
        tf.expand_dims(layer_outputs, 2), config.num_classes,
        (1, 1), name='output_linear_trans')  # [B, T, 1, F]
    outputs = tf.squeeze(outputs, 2)  # [B, T, F]
    if config.use_relu:
        outputs = tf.nn.relu(outputs)
    return outputs


class DRNN(object):
    def __init__(self, config, input, is_train):
        self.config = config
        if is_train:
            stager, self.stage_op, self.input_filequeue_enqueue_op = input
            # we only use 1 gpu
            self.inputX, self.label_values, self.label_indices, self.label_dense_shape, self.seqLengths = stager.get()
            self.label_batch = tf.SparseTensor(
                self.label_indices,
                tf.cast(self.label_values, tf.int32),
                self.label_dense_shape)
            self.build_graph(config, is_train)
        else:
            stager, self.stage_op, self.input_filequeue_enqueue_op = input
            self.inputX, self.seqLengths, self.correctness, self.labels, self.names = stager.get()
            self.build_graph(config, is_train)

    @describe
    def build_graph(self, config, is_train):

        self.nn_outputs = inference(self.inputX, self.seqLengths, config)
        self.ctc_input = tf.transpose(self.nn_outputs, perm=[1, 0, 2])

        if is_train:
            self.label_dense = tf.sparse_tensor_to_dense(self.label_batch)
            self.ctc_loss = tf.nn.ctc_loss(inputs=self.ctc_input,
                                           labels=self.label_batch,
                                           sequence_length=self.seqLengths,
                                           ctc_merge_repeated=True,
                                           preprocess_collapse_repeated=False,
                                           time_major=True)
            self.loss = tf.reduce_sum(self.ctc_loss) / config.batch_size
            self.global_step = tf.Variable(0, trainable=False)
            self.reset_global_step = tf.assign(self.global_step, 1)

            initial_learning_rate = tf.Variable(
                config.learning_rate, trainable=False)

            self.learning_rate = tf.train.exponential_decay(
                initial_learning_rate, self.global_step, self.config.decay_step,
                self.config.lr_decay, name='lr')

            self.warmup = 20000
            self.sqrtstep = tf.sqrt(tf.cast(self.global_step, tf.float32))
            self.learning_rate = 0.2 * tf.minimum(
                1 / self.sqrtstep, tf.div(self.sqrtstep, self.warmup))

            if config.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif config.optimizer == 'nesterov':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                            0.9,
                                                            use_nesterov=True)
            else:
                raise Exception('optimizer not defined')

            self.vs = tf.trainable_variables()
            grads_and_vars = self.optimizer.compute_gradients(self.loss,
                                                              self.vs)
            self.grads = [grad for (grad, var) in grads_and_vars]
            self.vs = [var for (grad, var) in grads_and_vars]
            if config.max_grad_norm > 0:
                self.grads, _ = tf.clip_by_global_norm(
                    self.grads, config.max_grad_norm)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.vs),
                global_step=self.global_step)
        else:
            self.softmax = tf.nn.softmax(self.ctc_input)
            self.ctc_decode_input = tf.log(self.softmax)
            self.ctc_decode_result, self.ctc_decode_log_prob = tf.nn.ctc_beam_search_decoder(
                self.ctc_decode_input, self.seqLengths,
                beam_width=config.beam_size, top_paths=1)
            self.dense_output = tf.sparse_tensor_to_dense(
                self.ctc_decode_result[0], default_value=-1)


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
            attention_output = inference(inputX, self.seqLength, config)

            with tf.name_scope('fc-layer'):
                if config.use_project:
                    weightsClasses = tf.get_variable(name='weightsClasses',
                                                     initializer=tf.truncated_normal(
                                                         [config.num_proj,
                                                          config.num_classes]))
                    flatten_outputs = tf.reshape(attention_output,
                                                 (-1, config.num_proj))
                else:
                    weightsClasses = tf.get_variable(name='weightsClasses',
                                                     initializer=tf.truncated_normal(
                                                         [config.hidden_size,
                                                          config.num_classes]))
                    flatten_outputs = tf.reshape(attention_output,
                                                 (-1, config.hidden_size))
                biasesClasses = tf.get_variable(name='biasesClasses',
                                                initializer=tf.truncated_normal(
                                                    [config.num_classes]))

            flatten_logits = tf.matmul(flatten_outputs,
                                       weightsClasses) + biasesClasses
            self.softmax = tf.nn.softmax(flatten_logits, name='softmax')


if __name__ == "__main__":
    pass
