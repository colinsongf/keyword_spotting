import numpy as np
import tensorflow as tf
import librosa
from mfcc import mfcc_op

y, sr = librosa.load('./temp1.wav')
librosa.feature.mfcc()
spec = np.abs(librosa.stft(y, 400, 160, 400)) ** 2
print(spec.shape)

size = 16
# length = tf.placeholder(tf.float32,shape=[None,])
#
#
# with tf.Session() as session:
#     ret, = session.run([pe], {length: 10})
#     print(ret)
