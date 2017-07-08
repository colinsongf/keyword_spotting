import numpy as np
import tensorflow as tf
import librosa
# from mfcc import mfcc_op
from python_speech_features import mfcc,delta

y, sr = librosa.load('./temp1.wav', sr=16000)

def pre_emphasis(signal,coefficient=0.97):
    '''对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return np.append(signal[0],signal[1:]-coefficient*signal[:-1])
y=np.asarray([1,1,1,4,1,1,1,])
y1 = pre_emphasis(y)
# librosa.output.write_wav('./temp2.wav',y,16000)
print(y1)
print(y)
print('done')

# spec = np.abs(librosa.stft(y, 400, 160, 400)) ** 2
# print(spec.shape)

size = 16
# length = tf.placeholder(tf.float32,shape=[None,])
#
#
# with tf.Session() as session:
#     ret, = session.run([pe], {length: 10})
#     print(ret)
