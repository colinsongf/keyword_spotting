# encoding: utf-8

'''

@author: ZiqiLiu


@file: stft.py

@time: 2017/7/11 下午12:47

@desc:
'''
import tensorflow as tf
import librosa
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops



# tensorflow implement of frame (tf r1.1), actually in tf 1.3 there is a new implement, which is better
def tf_frame(signal, frame_length, frame_step, name=None):
    """Frame a signal into overlapping frames.
    May be used in front of spectral functions.
    For example:
    ```python
    pcm = tf.placeholder(tf.float32, [None, 9152])
    frames = tf.contrib.signal.frames(pcm, 512, 180)
    magspec = tf.abs(tf.spectral.rfft(frames, [512]))
    image = tf.expand_dims(magspec, 3)
    ```
    Args:
      signal: A `Tensor` of shape `[batch_size, signal_length]`.
      frame_length: An `int32` or `int64` `Tensor`. The length of each frame.
      frame_step: An `int32` or `int64` `Tensor`. The step between frames.
      name: A name for the operation (optional).
    Returns:
      A `Tensor` of frames with shape `[batch_size, num_frames, frame_length]`.
    Raises:
      ValueError: if signal does not have rank 2.
    """
    with ops.name_scope(name, "frames", [signal, frame_length, frame_step]):
        signal = ops.convert_to_tensor(signal, name="signal")
        frame_length = ops.convert_to_tensor(frame_length, name="frame_length")
        frame_step = ops.convert_to_tensor(frame_step, name="frame_step")

        signal_rank = signal.shape.ndims

        if signal_rank != 2:
            raise ValueError(
                "expected signal to have rank 2 but was " + signal_rank)

        signal_length = array_ops.shape(signal)[1]

        num_frames = math_ops.floor((signal_length - frame_length) / frame_step)
        num_frames = 1 + math_ops.cast(num_frames, dtypes.int32)

        # crop_len = frame_length+frame_step*(num_frames-1)
        # cropped_signal = tf.slice(signal,[0,0],[-1,crop_len]);


        indices_frame = array_ops.expand_dims(math_ops.range(frame_length), 0)
        indices_frames = array_ops.tile(indices_frame, [num_frames, 1])

        indices_step = array_ops.expand_dims(
            math_ops.range(num_frames) * frame_step, 1)
        indices_steps = array_ops.tile(indices_step, [1, frame_length])

        indices = indices_frames + indices_steps

        # TODO(androbin): remove `transpose` when `gather` gets `axis` support
        signal = array_ops.transpose(signal)
        signal_frames = array_ops.gather(signal, indices)
        signal_frames = array_ops.transpose(signal_frames, perm=[2, 0, 1])

        return signal_frames



# this is only to simulate frame in tensorflow
def frame(y, n_fft=400, hop_length=160, win_length=400, window='hann'):
    if win_length is None:
        win_length = n_fft

        # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = librosa.filters.get_window(window, win_length, fftbins=True)
    print(fft_window.shape)

    # Pad the window out to n_fft size
    fft_window = librosa.util.pad_center(fft_window, n_fft)
    print(fft_window.shape)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered

    # Window the time series.
    y_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)

    return (y_frames * fft_window).T

