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
from utils.stft import tf_frame
from utils.custom_wrapper import LayerNormalizer, ResidualWrapper, \
    HighwayWrapper
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell
import librosa

cell_fn = GRUCell


# cell_fn = LSTMCell


class GRU(object):
    def __init__(self, config, input, is_train):
        self.config = config
        if is_train:
            stager, self.stage_op, self.input_filequeue_enqueue_op = input
            # we only use 1 gpu
            self.inputX, self.label_values, self.label_indices, self.label_dense_shape, self.seqLengths = stager.get()
            if config.keep_prob < 1:
                self.inputX = tf.nn.dropout(self.inputX, config.keep_prob)
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

        self.nn_outputs, _ = inference1(config, self.inputX,
                                        self.seqLengths, is_train)
        self.fc_outputs = inference2(self.nn_outputs, config)
        self.ctc_input = tf.transpose(self.fc_outputs, perm=[1, 0, 2])

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
            grads_and_vars = [(g, v) for g, v in grads_and_vars if
                              g is not None]
            self.grads = [grad for (grad, var) in grads_and_vars]
            self.vs = [var for (grad, var) in grads_and_vars]
            if config.max_grad_norm > 0:
                self.grads, hehe = tf.clip_by_global_norm(
                    self.grads, config.max_grad_norm)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.vs),
                global_step=self.global_step)
        else:
            self.ctc_input = tf.transpose(self.ctc_input, perm=[1, 0, 2])
            self.softmax = tf.nn.softmax(self.ctc_input, name='softmax')
            # self.ctc_decode_input = tf.log(self.softmax)
            # self.ctc_decode_result, self.ctc_decode_log_prob = tf.nn.ctc_beam_search_decoder(
            #     self.ctc_decode_input, self.seqLengths,
            #     beam_width=config.beam_size, top_paths=1)
            # self.dense_output = tf.sparse_tensor_to_dense(
            #     self.ctc_decode_result[0], default_value=-1)


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
            self._rnn_initial_states = tf.placeholder(
                tf.float32,
                [config.num_layers,
                 1,
                 config.hidden_size],
                name='rnn_initial_states')
            # self.inputX = tf.placeholder(dtype=tf.float32,
            #                              shape=[None, ],
            #                              name='inputX')
            #
            # self.inputX = tf.expand_dims(self.inputX, 0)
            # self.frames = tf_frame(self.inputX, 400, 160, name='frame')
            #
            # self.linearspec = tf.abs(tf.spectral.rfft(self.frames, [400]))
            #
            # self.mel_basis = librosa.filters.mel(
            #     sr=config.samplerate,
            #     n_fft=config.fft_size,
            #     fmin=config.fmin,
            #     fmax=config.fmax,
            #     n_mels=config.freq_size).T
            # self.mel_basis = tf.constant(value=self.mel_basis, dtype=tf.float32)
            # self.mel_basis = tf.expand_dims(self.mel_basis, 0)
            #
            # self.melspec = tf.matmul(self.linearspec, self.mel_basis,
            #                          name='mel')
            self.inputX = tf.placeholder(dtype=tf.float32,
                                         shape=[None, config.freq_size],
                                         name='inputX')
            self.melspec = tf.expand_dims(self.inputX, 0)

            self.seqLengths = tf.expand_dims(tf.shape(self.melspec)[1], 0)
            rnn_initial_states = tuple(tf.unstack(self._rnn_initial_states))
            self.nn_outputs, rnn_states = inference1(config, self.melspec,
                                                     self.seqLengths,
                                                     is_training=False,
                                                     initial_state=rnn_initial_states)
            self.rnn_states = tf.stack(rnn_states, name="rnn_states")
            self.linear_output = inference2(self.nn_outputs, config, 1)
            self.logits = tf.identity(self.linear_output,name='logit')

            self.softmax = tf.nn.softmax(self.linear_output, name='softmax')
            # ctc_decode_input = tf.log(self.softmax)
            # self.ctc_decode_input = tf.concat(
            #     [self._prev_ctc_decode_inputs, ctc_decode_input], axis=1,
            #     name="ctc_decode_inputs")
            # self.ctc_decode_result, self.ctc_decode_log_prob = tf.nn.ctc_beam_search_decoder(
            #     self.ctc_decode_input, self.seqLengths,
            #     beam_width=config.beam_size, top_paths=1)
            # self.dense_output = tf.sparse_tensor_to_dense(
            #     self.ctc_decode_result[0], default_value=-1,
            #     name='dense_output')


def get_cell(config, is_training):
    print(tf.get_variable_scope().reuse)
    cell = cell_fn(num_units=config.hidden_size,
                   # kernel_initializer=tf.contrib.layers.xavier_initializer,
                   activation=tf.tanh,
                   reuse=tf.get_variable_scope().reuse
                   )
    # add wrappers: ln -> dropout -> residual
    if config.use_layer_norm:
        cell = LayerNormalizer(cell)
    if is_training:
        if config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 output_keep_prob=config.keep_prob,
                                                 dtype=tf.float32,
                                                 variational_recurrent=config.variational_recurrent
                                                 )
    if config.use_residual:
        cell = ResidualWrapper(cell)

    return cell


def inference1(config,
               inputX,
               seqLengths, is_training, initial_state=None):
    """
       The model is splited into two parts in order to support streaming.
       This part contains the CNN layer and RNN layers in the model.
       input:
         inputs: a 3D Tensor with shape [batch_size, time_step, freq_size]
                 the input spectrum after stft and padding
         actual_lengths: a 1D Tensor with shape [batch_size] which indictates
                         the actual lengths of each instances in inputs
         initial_states: a list of Tensor with length of rnn_layer_nums,
                         each Tensor has the shape [batch_size, hidden_size]
                         the initial states of rnn cell
         global_step: a Tensorflow Variable which indicates the number of
                      step in training
         config: a ModelConfig instance which indicates the model config
                 see speech/attention_config.py
         is_train: a bool indicates whether in training
       return:
         rnn_outputs: a 3D Tensor with shape [time_step, batch_size, hidden_size]
                      the outputs of the last rnn layer
         rnn_states: a list of Tensor with length of rnn_layer_nums,
                     each Tensor has the shape [batch_size, hidden_size]
                     the final states of rnn cell
    """
    rnn_cells = []
    for i in range(config.num_layers):
        with tf.variable_scope('rnn_cell%d' % i,
                               initializer=tf.contrib.layers.xavier_initializer(
                                   uniform=False)):
            print('building RNN layer')
            rnn_cells.append(get_cell(config, is_training))

    rnn_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells)

    outputs, states = dynamic_rnn(rnn_cells,
                                  inputs=inputX,
                                  sequence_length=seqLengths,
                                  initial_state=initial_state,
                                  dtype=tf.float32,
                                  scope="drnn")
    return outputs, states


def inference2(rnn_outputs, config, batch_size=None):
    """
      The model is splited into two parts in order to support streaming.
      This part contains the lookahead layer and the full connect layer.
      input:
        rnn_outputs: a 3D Tensor with shape [time_step, batch_size, hidden_size]
                     the outputs of the last rnn layer
        config: a ModelConfig instance which indicates the model config
                see speech/attention_config.py
        is_train: a bool indicates whether in training
      return:
        linear_outputs: a 3D Tensor with shape
                        [batch_size,time_step,  num_tokens+1]
                        the outputs of the full connect layer
      """
    if batch_size is None:
        batch_size = config.batch_size
    with tf.name_scope('fc-layer'):
        weightsClasses = tf.get_variable(name='weightsClasses',
                                         initializer=tf.truncated_normal(
                                             [config.hidden_size,
                                              config.num_classes]))
        flatten_outputs = tf.reshape(rnn_outputs,
                                     (-1, config.hidden_size))
        biasesClasses = tf.get_variable(name='biasesClasses',
                                        initializer=tf.zeros(
                                            [config.num_classes]))

    flatten_logits = tf.add(tf.matmul(flatten_outputs,
                                      weightsClasses), biasesClasses,
                            name='linear_add')
    logits = tf.reshape(flatten_logits,
                        [batch_size, -1, config.num_classes],name='test')
    if config.use_relu:
        logits = tf.nn.relu(logits, name='relu')
        if config.value_clip > 0:
            logits = tf.clip_by_value(logits, 0, 20)
    return logits


if __name__ == "__main__":
    pass
