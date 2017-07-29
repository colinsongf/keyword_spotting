# encoding: utf-8

'''

@author: ZiqiLiu

This file contains custom wrapper for different cells

@file: custom_wrapper.py

@time: 2017/7/20 上午10:16

@desc:
'''
from __future__ import absolute_import
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
from tensorflow.python.ops import array_ops
import tensorflow as tf


class RNNWrapper(RNNCell):
    @property
    def input_size(self):
        return self._cell.input_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size


class HighwayWrapper(RNNWrapper):
    """Implementing https://arxiv.org/pdf/1505.00387v2.pdf, need to making
    sure this is shared from all layers.
    """

    def __init__(self, cell, weights):
        self._cell = cell
        self._weights = weights

    def __call__(self, inputs, state, scope=None):
        t = tf.sigmoid(tf.matmul(inputs, self._weights))
        output, new_state = self._cell(inputs, state, scope)
        output = output * t + (1.0 - t) * inputs
        return output, new_state


class ClockworkWrapper(RNNWrapper):
    """
    Operator adding clockwork to state of the given cell.
    (cf. Koutnik et al. 2014 [arXiv, https://arxiv.org/abs/1402.3511]).
    """

    def __init__(self, cell, period):
        """
        Create a cell with clockwork.
        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          period: `period` number of states will be a transfer group
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if period is not an int.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if not isinstance(period, int):
            raise ValueError("Parameter period must be an integer: %d" %
                             period)
        self._cell = cell
        self._period = period
        self._state_buffer = None
        self._buffer_index = period - 1

    def __call__(self, inputs, state_in, scope=None):
        # in 0-period step, state is the initial state.
        # from period step, state is fetched from state_buffer.
        if not self._state_buffer:
            input_shape = array_ops.shape(inputs)
            batch_size = input_shape[0]
            initial_state = self._cell.zero_state(
                batch_size, dtype=inputs.dtype)
            self._state_buffer = [initial_state] * self._period

        self._state_buffer[self._buffer_index] = state_in
        self._buffer_index = (self._buffer_index + 1) % self._period
        state = self._state_buffer[self._buffer_index]

        output, new_state = self._cell(inputs, state, scope)
        return output, new_state


class ResidualWrapper(RNNWrapper):
    """
    This add residual support to any rnn: simply add input to output.
    """

    def __init__(self, cell):
        """
        Create a cell with residue.
        Args:
          cell: an RNNCell, a projection to output_size is added to it.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the clockwork state."""
        output, new_state = self._cell(inputs, state, scope)
        return 0.7071067811865475 * (output + inputs), new_state


class LayerNormalizer(RNNWrapper):
    """
    This add layer normalization support to any rn cell:
      do normalization on input or state.
    """

    @staticmethod
    def _ln(input, s, b, epsilon=1e-5):
        """ Layer normalizes a 2D tensor [b,f]"""
        m, v = tf.nn.moments(input, [1], keep_dims=True)
        normalised_input = (input - m) / tf.sqrt(v + epsilon)
        return normalised_input * s + b

    def __init__(self, cell):
        """
        Create a cell with layer normalization.
        Args:
          cell: an RNNCell, a projection to output_size is added to it.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the layer normalization."""
        with tf.variable_scope(type(self).__name__), tf.device('/cpu:0'):
            self.ibeta = tf.get_variable(
                initializer=tf.constant(0.0, shape=[]),
                name='ibeta', trainable=True)
            self.igamma = tf.get_variable(
                initializer=tf.constant(1.0, shape=[inputs.get_shape()[1]]),
                name='igamma', trainable=True)

        inputs = LayerNormalizer._ln(inputs, self.ibeta, self.igamma)

        output, new_state = self._cell(inputs, state, scope)
        return output, new_state
