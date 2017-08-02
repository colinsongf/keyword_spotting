from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from .op_compile import OperaterCompiler

compiler = OperaterCompiler('Octbit', osp.dirname(osp.abspath(__file__)))

compiler.record_cpu_basis(['octbit_mat_mul_op.cc', 'octbit_ops_reg.cc'],
                          '_octbit_ops.so')
_octbit_ops_so = compiler.compile()

assert _octbit_ops_so, "Could not load _octbit_ops.so."


def octbit_mat_mul(x1, x2,
                   transpose_a=False, transpose_b=True,
                   scale=0.0, bias=[0]):
    return _octbit_ops_so.octbit_mat_mul(
        x1,
        x2,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        scale=scale,
        bias=bias)


# TODO: add shape registration
