import os
import tensorflow as tf


class OperaterCompiler:
    def __init__(self, op_name, source_dir, lib_dirs=None):
        self._op_name = op_name
        self._source_dir = source_dir
        self._lib_dirs = lib_dirs if lib_dirs else []

    def record_cpu_basis(self, cc_paths, so_path, ext=''):
        self._cc_paths = [
            os.path.join(self._source_dir, path) for path in cc_paths
        ]
        self._so_path = os.path.join(self._source_dir, so_path)
        print(self._cc_paths)
        print(self._so_path)
        self._cpu_ext = ext
        self._cucc_paths = []

    def record_gpu_kernel_builders(self, cucc_paths, ext=''):
        self._cucc_paths = [
            os.path.join(self._source_dir, path) for path in cucc_paths
        ]
        self._gpu_ext = ext

    def compile(self):
        # remove existing library:
        if os.path.exists(self._so_path):
            os.remove(self._so_path)

        # check if gpu is available
        from tensorflow.python.client import device_lib
        gpu_available = any(
            device.device_type == 'GPU'
            for device in device_lib.list_local_devices()
        )

        if gpu_available and self._cucc_paths:
            cuo_paths = [path[:-3] + '.o' for path in self._cucc_paths]
            print('Compiling {} GPU kernel...'.format(self._op_name))

            for cucc_path, cuo_path in zip(self._cucc_paths, cuo_paths):
                os.system(
                    'nvcc -std=c++11 -c %s -o %s -D_MWAITXINTRIN_H_INCLUDED '
                    '-I %s %s -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -w '
                    '-D_GLIBCXX_USE_CXX11_ABI=0 %s' % (
                        cucc_path,
                        cucc_path[:-3] + '.o',  # .cu.o path
                        tf.sysconfig.get_include(),
                        ' '.join(
                            ['-I ' + lib_dir for lib_dir in self._lib_dirs]
                        ),
                        self._gpu_ext
                    ))

        # if gpu is not available, only compile CPU kernels
        else:
            cuo_paths = []

        print('Compiling {} CPU kernel...'.format(self._op_name))
        print('.so path', self._so_path)
        print('lib dirs', self._lib_dirs)
        os.system(
            'g++ -std=c++11 -shared %s -o %s -fPIC -I %s %s -O2 '
            '-D_GLIBCXX_USE_CXX11_ABI=0 -msse4.1 -w %s' % (
                ' '.join(self._cc_paths + cuo_paths),
                self._so_path,
                tf.sysconfig.get_include(),
                ' '.join(['-I ' + lib_dir for lib_dir in self._lib_dirs]),
                self._cpu_ext
            ))

        if gpu_available:
            for cuo_path in cuo_paths:
                if os.path.exists(cuo_path):
                    os.remove(cuo_path)

        return tf.load_op_library(self._so_path)
