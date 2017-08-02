from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import plugins.octbit.octbit_ops as octbit_ops
from tensorflow.python.framework import tensor_util


class OctbitMatmulTest(tf.test.TestCase):
    _use_gpu = True

    def testOctbitMatmul(self):
        _np_qint8 = np.dtype([("qint8", np.int8, 1)])
        with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()):
            # first test case
            # x1: [[-1., -1., ..., -1.]]
            # x2: [[0, 1, 2, ..., 63]]
            # sclae: 3.0
            # result: [[-6048.]]
            #
            x1 = [[-1. for i in range(64)]]
            x2 = np.array([[i for i in range(64)]], dtype=_np_qint8[0])
            bias = [127 * 2016.0]
            bias = tensor_util.make_tensor_proto(bias,
                                                 dtype=tf.float32,
                                                 shape=None)

            result = octbit_ops.octbit_mat_mul(x1, x2, scale=3.0, bias=bias)
            self.assertAllEqual(
                result.eval(),
                [[-6048.]])
            # seconde test case
            # x1: [[-1, -1, ..., -1], [-1, -1, ..., -1]]
            # x2: [[1, 1, ..., 1], [0, 1, 2, ..., 63] * 3]
            # scale: 2.0
            # result: [[-128., -4032., -4032., -4032.],
            #          [-128., -4032., -4032., -4032.]]
            x1 = [[-1 for i in range(64)] for j in range(2)]
            x2 = [[1 for i in range(64)]]
            x2 = x2 + [[i for i in range(64)] for j in range(3)]
            x2 = np.array(x2, dtype=_np_qint8[0])
            bias = [127 * 64., 127 * 2016., 127 * 2016., 127 * 2016.]
            bias = tensor_util.make_tensor_proto(bias,
                                                 dtype=tf.float32,
                                                 shape=None)
            result = octbit_ops.octbit_mat_mul(x1, x2, scale=2.0, bias=bias)
            self.assertAllEqual(
                result.eval(),
                [[-128., -4032., -4032., -4032.],
                 [-128., -4032., -4032., -4032.]])


if __name__ == "__main__":
    tf.test.main()
