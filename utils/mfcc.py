# encoding: utf-8

'''

@author: ZiqiLiu


@file: mfcc.py

@time: 2017/7/8 下午10:18

@desc:
'''

import tensorflow as tf
import numpy as np
import librosa


def power_to_db(S, amin=1e-10, top_db=80.0):
    # S must be real number (magnitude)
    # S.shape = (B,T,H)
    log_spec = 10.0 * tf.log(tf.maximum(amin, S)) / tf.log(10.0)
    if top_db is not None:
        if top_db < 0:
            raise Exception('top_db must be non-negative')
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
    return log_spec


def dct(n_filters, n_input):
    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)

    return basis.T


def delta(feat, N):
    from functools import reduce
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_rersult = reduce(lambda a, b: a + b,
                           [_delta_order(feat, i) for i in
                            range(1, 1 + N)])
    delta_rersult = delta_rersult / denominator

    return delta_rersult


def _delta_order(feat, order):
    tf.pad(feat, [[0, 0], [2 * order, 2 * order], [0, 0]], "CONSTANT")
    padding_head = tf.slice(feat, [0, 0, 0], [-1, 1, -1])
    padding_head = tf.tile(padding_head, [1, 2, 1])
    padding_tail = tf.slice(feat, [0, tf.shape(feat)[1] - 1, 0], [-1, 1, -1])
    padding_tail = tf.tile(padding_tail, [1, 2, 1])

    subtracted = tf.concat([feat, padding_tail], 1)
    subtractor = tf.concat([padding_head, feat], 1)
    delta_result = (subtracted - subtractor) * order

    return tf.slice(delta_result, [0, 1, 0], [-1, tf.shape(feat)[1], -1])


def mfcc(linearspec, config, n_mfcc=13, top_db=None):
    # linearspec.shape=(T,B,H)

    linearspec = tf.square(linearspec)
    mel_basis = librosa.filters.mel(
        sr=config.samplerate,
        n_fft=config.fft_size,
        fmin=config.fmin,
        fmax=config.fmax,
        n_mels=config.n_mel).T
    mel_basis = tf.constant(value=mel_basis, dtype=tf.float32)
    mel_basis = tf.tile(tf.expand_dims(mel_basis, 0),
                        [config.batch_size, 1, 1])
    melspec = tf.matmul(linearspec, mel_basis)
    S = power_to_db(melspec, 1e-10, top_db)

    dct_basis = dct(n_mfcc, config.n_mel)
    dct_basis = tf.tile(tf.expand_dims(dct_basis, 0),
                        [config.batch_size, 1, 1])
    dct_basis = tf.cast(dct_basis, tf.float32)

    mfcc = tf.matmul(S, dct_basis)
    mfcc_first_order = delta(mfcc, 1)
    mfcc_second_order = delta(mfcc, 2)
    tf.concat([mfcc, mfcc_first_order], 2)
    return tf.concat([mfcc, mfcc_first_order, mfcc_second_order], 2)
