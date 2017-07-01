# -*- coding: utf-8 -*-
# Copyright © 2017 Naturali, Inc.
# All rights reserved.
# Created by Liu Jiahua (jiahua.liu@naturali.io) on 16 Jun 2017

import os

from .op_compile import OperaterCompiler

source_dir = os.path.dirname(os.path.abspath(__file__))
lib_dirs = None


compiler = OperaterCompiler('Positional Encoding',
                            source_dir,
                            lib_dirs)

compiler.record_cpu_basis(['positional_encoding_op.cc'],
                          '_positional_encoding_op.so')

_positional_encoding_module = compiler.compile()

def positional_encoding(max_position, encoding_size):
    return _positional_encoding_module.positional_encoding(max_position, encoding_size)

