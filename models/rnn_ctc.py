# encoding: utf-8

'''

@author: ZiqiLiu


@file: rnn_ctc.py

@time: 2017/5/18 上午11:04

@desc:
'''

# !/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
from utils.common import describe
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell

cell_fn = GRUCell


# cell_fn = LSTMCell


class GRU(object):
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
            self.inputX, self.seqLengths, self.correctness, self.labels = stager.get()
            self.build_graph(config, is_train)

    @describe
    def build_graph(self, config, is_train):

        self.nn_outputs = build_dynamic_rnn(config, self.inputX,
                                            self.seqLengths, is_train)
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
            if config.warmup:
                self.warmup_lr = tf.train.polynomial_decay(5e-3,
                                                           self.global_step,
                                                           40000, 1.35e-3, 0.5)
                self.post_lr = tf.train.exponential_decay(
                    1.5e-3, self.global_step, self.config.decay_step,
                    self.config.lr_decay, name='lr')
                self.learning_rate = tf.minimum(self.warmup_lr, self.post_lr)

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
        config.use_bg_noise = False
        config.use_white_noise = False
        config.keep_prob = 1

        with tf.device('/cpu:0'):
            self.inputX = tf.placeholder(dtype=tf.float32,
                                         shape=[None, config.num_features],
                                         name='inputX')
            self.inputX = tf.expand_dims(self.inputX, 0, name='reshape_inputX')
            self.fuck = tf.identity(self.inputX, name='fuck')

            self.seqLengths = tf.placeholder(dtype=tf.int32, shape=[1],
                                             name='seqLength')
            self.nn_outputs = build_dynamic_rnn(config, self.inputX,
                                                self.seqLengths,
                                                is_training=False)
            self.ctc_input = tf.transpose(self.nn_outputs, perm=[1, 0, 2])

            self.softmax = tf.nn.softmax(self.ctc_input)
            self.ctc_decode_input = tf.log(self.softmax)
            self.ctc_decode_result, self.ctc_decode_log_prob = tf.nn.ctc_beam_search_decoder(
                self.ctc_decode_input, self.seqLengths,
                beam_width=config.beam_size, top_paths=1)
            self.dense_output = tf.sparse_tensor_to_dense(
                self.ctc_decode_result[0], default_value=-1)


def get_cell(config, is_training):
    print(tf.get_variable_scope().reuse)
    cell = cell_fn(num_units=config.hidden_size,
                   # kernel_initializer=tf.contrib.layers.xavier_initializer,
                   activation=tf.tanh,
                   reuse=tf.get_variable_scope().reuse
                   )
    # cell = cell_fn(num_units=config.hidden_size,
    #                use_peepholes=True,
    #                cell_clip=config.cell_clip,
    #                initializer=tf.contrib.layers.xavier_initializer(),
    #                forget_bias=1.0,
    #                state_is_tuple=True,
    #                activation=tf.tanh,
    #                reuse=tf.get_variable_scope().reuse
    #                )
    if is_training:
        if config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 output_keep_prob=config.keep_prob,
                                                 dtype=tf.float32,
                                                 variational_recurrent=config.variational_recurrent
                                                 )

    return cell


def build_dynamic_rnn(config,
                      inputX,
                      seqLengths, is_training):
    with tf.variable_scope('rnn_cell',
                           initializer=tf.contrib.layers.xavier_initializer(
                               uniform=False)):
        if config.num_layers > 1:
            print('building multi layer LSTM')
            cell = tf.contrib.rnn.MultiRNNCell(
                [get_cell(config, is_training) for _ in
                 range(config.num_layers)])
        else:
            # cell = tf.contrib.rnn.MultiRNNCell([get_cell(config)])
            cell = get_cell(config, is_training)
    outputs, output_states = dynamic_rnn(cell,
                                         inputs=inputX,
                                         sequence_length=seqLengths,
                                         initial_state=None,
                                         dtype=tf.float32,
                                         scope="drnn")

    with tf.name_scope('fc-layer'):

        weightsClasses = tf.get_variable(name='weightsClasses',
                                         initializer=tf.truncated_normal(
                                             [config.hidden_size,
                                              config.num_classes]))
        flatten_outputs = tf.reshape(outputs,
                                     (-1, config.hidden_size))
        biasesClasses = tf.get_variable(name='biasesClasses',
                                        initializer=tf.zeros(
                                            [config.num_classes]))

    flatten_logits = tf.matmul(flatten_outputs,
                               weightsClasses) + biasesClasses
    logits = tf.reshape(flatten_logits,
                        [config.batch_size, -1, config.num_classes])
    return logits


if __name__ == "__main__":
    pass
