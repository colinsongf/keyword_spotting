# -*- coding: utf-8 -*-
# Copyright © 2017 Naturali, Inc.
# All rights reserved.
# Created by Liu Jiahua (jiahua.liu@naturali.io) on 16 Jun 2017

import os

from .op_compile import OperaterCompiler

source_dir = os.path.dirname(os.path.abspath(__file__))
lib_dirs = None

compiler = OperaterCompiler('mfcc',
                            source_dir,
                            lib_dirs)

compiler.record_cpu_basis(['mfcc_op.cc'],
                          '_mfcc_op.so')

_mfcc_module = compiler.compile()


def mfcc(spectrogram, sample_rate, upper_frequency_limit,
         lower_frequency_limit, filterbank_channel_count,
         dct_coefficient_count):
    return _mfcc_module.mfcc(spectrogram, sample_rate, upper_frequency_limit,
                             lower_frequency_limit, filterbank_channel_count,
                             dct_coefficient_count)
