# encoding: utf-8

'''

@author: ZiqiLiu


@file: server_demo.py

@time: 2017/6/14 下午5:04

@desc:
'''
import os
import asyncio
import matplotlib.pyplot as plt
import matplotlib
import argparse

matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import tornado
import tornado.web
from tornado.platform.asyncio import AsyncIOMainLoop
from positional_encoding import positional_encoding_op
import numpy as np
import tensorflow as tf
# from config.config import get_config
from config.attention_config import get_config
from utils.common import path_join
from utils.prediction import  ctc_predict, ctc_decode, \
    ctc_decode_strict
from fetch_wave import fetch
from normalize import main
import librosa

# load graph

config = get_config()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def process_wave(f):
    y, sr = librosa.load(f, sr=config.samplerate)
    stft = librosa.stft(y, 400, 160, 400)

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size,
                                       hop_length=config.step_size,
                                       power=config.power,
                                       fmin=300,
                                       fmax=8000,
                                       n_mels=60))

    return mel_spectrogram, y


def frame(y, n_fft=400, hop_length=160, win_length=400, window='hann'):
    if win_length is None:
        win_length = n_fft

        # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = librosa.util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered

    y = np.pad(y, int(n_fft // 2), mode='reflect')

    # Window the time series.
    y_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)

    return (y_frames * fft_window).T


class Runner():
    def __init__(self, config):
        self.graph_def = tf.GraphDef()
        self.config = config
        with open(path_join(config.graph_path, config.graph_name), 'rb') as f:
            # print (f.read())
            self.graph_def.ParseFromString(f.read())
        # for node in self.graph_def.node:
        #     print(node.name)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")
        nodes = [node.name for node in self.graph.get_operations()]
        for i in nodes:
            if 'nn' in i:
                print(i)
        self.sess = tf.Session(graph=self.graph)

    def predict_ctc(self, inputX):
        # with self.sess as sess, self.graph.as_default():
        st = time.time()
        softmax1, softmax2, nnoutput = self.sess.run(
            ['model/softmax1:0', 'model/softmax2:0', 'model/nn_outputs:0'],
            feed_dict={'model/inputX:0': inputX})
        end = time.time()
        print('processing time', end - st)

        colors = ['r', 'b', 'g', 'm', 'y', 'k', 'r', 'b']
        y = softmax2
        x = range(len(y))
        print(y.shape)
        plt.figure(figsize=(10, 4))  # 创建绘图对象
        for i in range(1, y.shape[1]):
            plt.plot(x, y[:, i], colors[i - 1], linewidth=1, label=str(i))
        plt.legend(loc='upper right')
        plt.savefig('./fig.png')
        output1 = ctc_decode_strict(softmax1, config.num_classes)
        output2 = ctc_decode(softmax2, config.num_classes)
        np.set_printoptions(precision=4, threshold=np.inf,
                            suppress=True)
        # print(prob)
        # result = True if ctc_predict(output[0]) else False
        result = ctc_predict(output1, config.label_seqs) | ctc_predict(output2,
                                                                       config.label_seqs)
        return result, str(output1)+'---' + str(output2)


def run(device_id='32EFEA3263D079E1BE3767C87FC0A1C2', current=False):
    config = get_config()

    runner = Runner(config)
    label = ""
    if not current:
        wave, label = fetch(device_id)
        print('wave', wave)
    frames = frame('temp.wav')
    result = runner.predict_ctc(frames)

    print(result, label)


class HotWordHandler(tornado.web.RequestHandler):
    def initialize(self, runner):
        self.runner = runner

    def get(self):
        device_id = self.get_argument('device_id')
        wave, label, wave_id = fetch(device_id)
        print('wave', wave)
        main()

        y, _ = librosa.load('normalized-temp.wav', sr=config.samplerate)
        frames = frame(y)

        result, output = self.runner.predict_ctc(y)
        self.write({
            'result': result,
            'label': label,
            'time': time.strftime('%Y-%m-%d %A %X %Z',
                                  time.localtime(time.time())),
            'wave_id': wave_id,
            'output': output
        })


def start_server(port):
    runner = Runner(config)
    AsyncIOMainLoop().install()
    app = tornado.web.Application([
        (r'/()', tornado.web.StaticFileHandler, {
            'path': BASE_DIR,
            'default_filename': 'index.html'
        }),
        (r'/api/hotword', HotWordHandler, {'runner': runner}),
    ], debug=True
    )
    app.listen(port)
    print('start server')
    loop = asyncio.get_event_loop()
    loop.run_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port',
                        help='server port, 8080 by dedault',
                        type=int,
                        default=8080)
    flags = parser.parse_args().__dict__
    start_server(flags['port'])
    # run('32EFEA3263D079E1BE3767C87FC0A1C2', True)
