import numpy as np
import tensorflow as tf
import librosa
# from mfcc import mfcc_op
from python_speech_features import mfcc,delta

y, sr = librosa.load('./temp1.wav', sr=16000)
spec = np.abs(librosa.stft(y, 400, 160, 400)) ** 2
# mfcc1 = librosa.feature.mfcc(y, sr=16000, n_mfcc=13, n_fft=400, hop_length=160,
#                              power=2.0, n_mels=60, fmin=0.0, fmax=None)
mfcc1 = librosa.feature.mfcc(sr=16000,S=spec,n_mfcc=13)

mfcc2 = mfcc(y,16000,0.025,0.01,13,60,400,0,8000,1,-1)
print(mfcc2.shape)
print(mfcc1.shape)
d1 = delta(mfcc2,1)
print(d1.shape)
librosa.feature.delta
# spec = np.abs(librosa.stft(y, 400, 160, 400)) ** 2
# print(spec.shape)

size = 16
# length = tf.placeholder(tf.float32,shape=[None,])
#
#
# with tf.Session() as session:
#     ret, = session.run([pe], {length: 10})
#     print(ret)
