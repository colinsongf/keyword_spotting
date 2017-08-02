# encoding: utf-8

'''

@author: ZiqiLiu


@file: detector.py

@time: 2017/7/17 下午5:08

@desc:
'''
# !/usr/bin/env python

import collections
import logging
import os
import signal
import time
import wave

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import tensorflow as tf

from config.rnn_config import get_config
from utils.basic_vad import vad
from utils.prediction import ctc_predict, ctc_decode
from utils.queue import SimpleQueue

interrupted = False

config = get_config()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    scale = 1. / float(1 << ((8 * n_bytes) - 1))
    fmt = '<i{:d}'.format(n_bytes)
    return scale * np.frombuffer(x, fmt).astype(dtype)


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


def interrupt_callback():
    global interrupted
    return interrupted


logging.basicConfig()
logger = logging.getLogger("keyword spotting")
logger.setLevel(logging.INFO)
TOP_DIR = os.path.dirname(os.path.abspath(__file__))

DETECT_SOUND = os.path.join(TOP_DIR, "resources/goat1.wav")


class RingBuffer(object):
    """Ring buffer to hold audio from PortAudio"""

    def __init__(self, size=16000 * 5):
        self._buf = collections.deque(maxlen=size)

    def extend(self, data):
        """Adds data to the end of buffer"""
        self._buf.extend(data)

    def get(self):
        """Retrieves data from the beginning of buffer and clears it"""
        tmp = bytes(bytearray(self._buf))
        self._buf.clear()
        float32 = buf_to_float(np.fromstring(tmp, "Int16"))
        return float32


def play_audio_file(fname=DETECT_SOUND):
    """Simple callback function to play a wave file. By default it plays
    a Ding sound.

    :param str fname: wave file name
    :return: None
    """
    ding_wav = wave.open(fname, 'rb')
    ding_data = ding_wav.readframes(ding_wav.getnframes())
    audio = pyaudio.PyAudio()
    stream_out = audio.open(
        format=audio.get_format_from_width(ding_wav.getsampwidth()),
        channels=ding_wav.getnchannels(),
        rate=ding_wav.getframerate(), input=False, output=True)
    stream_out.start_stream()
    stream_out.write(ding_data)
    time.sleep(0.2)
    stream_out.stop_stream()
    stream_out.close()
    audio.terminate()


class HotwordDetector(object):
    def __init__(self, model_file='./graph/mel60/graph.pb'):

        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.extend(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue

        self.ring_buffer = RingBuffer()
        self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(
            input=True, output=False,
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            frames_per_buffer=3600,
            stream_callback=audio_callback)
        self.npdata = []
        self.prob_queue = SimpleQueue(15)
        self.state = np.zeros([config.num_layers, 1, config.hidden_size],
                              dtype=np.float32)
        self.res = np.zeros([0], np.float32)
        self.mel_basis = librosa.filters.mel(config.samplerate, config.fft_size,
                                             fmin=config.fmin,
                                             fmax=config.fmax,
                                             n_mels=config.freq_size).T
        self.seg_count = 0
        self.prev_speech = True
        self.logits = []

        self.graph_def = tf.GraphDef()
        self.config = config
        with open(model_file, 'rb') as f:
            # print (f.read())
            self.graph_def.ParseFromString(f.read())
        # for node in self.graph_def.node:
        #     print(node.name)

        self.sess = tf.Session()
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")
        self.sess = tf.Session(graph=self.graph)

    def start(self, detected_callback=play_audio_file,
              interrupt_check=lambda: False,
              sleep_time=0.3):

        if interrupt_check():
            logger.debug("detect voice return")
            return

        logger.debug("detecting...")
        with self.sess as sess:
            while True:
                if interrupt_check():
                    logger.debug("detect voice break")
                    break
                data = self.ring_buffer.get()
                self.npdata.append(data)
                if len(data) == 0:
                    time.sleep(sleep_time)
                    continue

                if vad(data, 30):
                    pass
                    self.prev_speech = True
                else:
                    # if self.prev_speech:
                    #     self.prev_speech = False
                    #     self.clean_state()
                    #     self.prob_queue.clear()
                    self.clean_state()
                    self.prob_queue.clear()

                data = np.concatenate((self.res, data), 0)

                res = (len(data) - config.fft_size) % config.hop_size + (
                    config.fft_size - config.hop_size)
                self.res = data[-res:]
                #
                # linearspec = np.abs(
                #     librosa.stft(data, config.fft_size, config.hop_size,
                #                  config.fft_size, center=False)).T
                # mel = np.dot(linearspec, self.mel_basis)

                softmax, state = sess.run(
                    ['model/softmax:0', 'model/rnn_states:0'],
                    feed_dict={'model/inputX:0': data,
                               'model/rnn_initial_states:0': self.state})

                self.prob_queue.add(softmax)
                self.state = state
                concated_soft = np.concatenate(self.prob_queue.get_all(), 0)
                print(concated_soft.shape)

                result = ctc_decode(concated_soft)
                if ctc_predict(result,'1233'):
                    detected_callback()
                    self.prob_queue.clear()
                    librosa.output.write_wav('./trigger.wav',
                                             np.concatenate(self.npdata, 0),
                                             16000)
                    self.npdata = []
                    self.clean_state()
                    self.plot(concated_soft, 'trigger.png')

        logger.debug("finished.")
        self.terminate()

    def test(self, name, detected_callback=play_audio_file):
        y, sr = librosa.load(name, 16000)
        linearspec = np.abs(
            librosa.stft(y, config.fft_size, config.hop_size,
                         config.fft_size, center=False)).T
        mel = np.dot(linearspec, self.mel_basis)
        softmax, logits, state = self.sess.run(
            ['model/softmax:0', 'model/logit:0', 'model/rnn_states:0'],
            feed_dict={'model/inputX:0': y,
                       'model/rnn_initial_states:0': self.state})
        result = ctc_decode(softmax)
        print(result)
        if ctc_predict(result,'1233'):
            detected_callback()
        colors = ['r', 'b', 'g', 'm', 'y', 'k']

        y = softmax
        np.set_printoptions(precision=4,
                            threshold=np.inf,
                            suppress=True)
        test = y.flatten()
        print(softmax.shape)
        print(test.tolist())
        x = range(len(y))
        plt.figure(figsize=(20, 15), dpi=120)  # 创建绘图对象
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        p1 = plt.subplot(211)
        p2 = plt.subplot(212)

        for i in range(0, y.shape[1]):
            p1.plot(x, y[:, i], colors[i], linewidth=2, label=str(i))
        p1.legend(loc='upper right', fontsize=20)
        # p1.figure(figsize=(20, 4))  # 创建绘图对象

        for i in range(0, logits.shape[2]):
            p2.plot(x, logits[0][:, i], colors[i], linewidth=2, label=str(i))
        p2.legend(loc='upper right', fontsize=20)
        plt.savefig('./test.png')

    def test2(self, name, detected_callback=play_audio_file):
        np.set_printoptions(precision=4,
                            threshold=np.inf,
                            suppress=True)
        y, sr = librosa.load(name, 16000)
        print(len(y))
        seg_len = 3600
        seg_num = len(y) // seg_len
        res = np.zeros([0], np.float32)
        origin_state = np.zeros([2, 1, 128], np.float32)
        data = []
        accu=np.zeros([0,6],np.float32)
        print(seg_num)
        for i in range(seg_num + 1):
            print(i)
            if i == seg_num:
                feed = y[i * seg_len:]
            else:
                feed = y[i * seg_len:(i + 1) * seg_len]
            data = np.concatenate((res, feed))
            reslen = (len(
                data) - self.config.fft_size) % self.config.hop_size + 240
            res = data[-reslen:]
            floatlen = len(data) - (len(
                data) - self.config.fft_size) % self.config.hop_size
            data = data[:floatlen]
            softmax, logits, state = self.sess.run(
                ['model/softmax:0', 'model/logit:0', 'model/rnn_states:0'],
                feed_dict={'model/inputX:0': data,
                           'model/rnn_initial_states:0': origin_state})
            origin_state = state
            accu = np.concatenate((accu,softmax),0)
            print(softmax.flatten().tolist())

        a=ctc_decode(accu)
        print(a)

    def plot(self, softmax, name='figure.png'):
        colors = ['r', 'b', 'g', 'm', 'y', 'k']
        y = softmax
        x = range(len(y))
        plt.figure(figsize=(10, 4))  # 创建绘图对象

        for i in range(1, y.shape[1]):
            plt.plot(x, y[:, i], colors[i], linewidth=1, label=str(i))
        plt.legend(loc='upper right')
        plt.savefig(name)

    def terminate(self):

        self.stream_in.stop_stream()
        self.stream_in.close()
        self.audio.terminate()
        import librosa
        librosa.output.write_wav('./temp.wav', np.concatenate(self.npdata, 0),
                                 16000)
        softmax = np.concatenate(self.prob_queue.get_all(), 0)
        self.plot(softmax)

    def clean_state(self):
        self.state = np.zeros([config.num_layers, 1, config.hidden_size],
                              dtype=np.float32)
        print('clean state')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    detector = HotwordDetector()
    print('Listening... Press Ctrl+C to exit')

    # main loop

    # detector.start(detected_callback=play_audio_file,
    #                interrupt_check=interrupt_callback,
    #                sleep_time=0.03)
    #


    detector.test('s_BA92843E74C1E7E6_你好乐乐.wav')
