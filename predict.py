# encoding: utf-8

'''

@author: ZiqiLiu


@file: predict.py

@time: 2017/6/14 下午5:04

@desc:
'''
import tensorflow as tf
from config.config import get_config
import argparse
from utils.common import path_join
from utils.prediction import moving_average, decode, predict
from process_wav import process_wave
import numpy as np

# load graph
class Runner():
    def __init__(self, config):
        self.graph_def = tf.GraphDef()
        self.config = config
        with open(path_join(config.working_path, config.graph_name), 'rb') as f:
            # print (f.read())
            self.graph_def.ParseFromString(f.read())
        # for node in self.graph_def.node:
        #     print(node.name)

        self.sess = tf.Session()
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")
        self.sess = tf.Session(graph=self.graph)

    def predict(self, inputX):
        seqLen = np.asarray([len(inputX)])
        with self.sess as sess, self.graph.as_default():

            # variable_names = [n.name for n in
            #                   sess.graph.as_graph_def().node]
            # for n in variable_names:
            #     print(n)
            prob = sess.run(['model/softmax:0'],
                            feed_dict={'model/inputX:0': inputX,
                                       'model/seqLength:0': seqLen})
            moving_avg = moving_average(prob[0], self.config.smoothing_window,
                                        padding=True)

            prediction = predict(moving_avg, self.config.trigger_threshold,
                                 self.config.lockout)
            result = decode(prediction, self.config.word_interval)
            print(result)


if __name__ == '__main__':
    config = get_config()
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='train: train model, ' +
                                       'valid: model validation, ',
                        default=None)
    parser.add_argument('-max', '--max_pooling_loss', help='1: maxpooling, ' +
                                                           '0: cross entropy,',
                        type=int,
                        default=None)
    parser.add_argument('-m', '--model_path',
                        help='The  model path for restoring',
                        default=None)
    parser.add_argument('-w', '--working_path',
                        help='The  model path for  saving',
                        default=None)
    parser.add_argument('-g', '--gpu',
                        help='visible GPU',
                        default=None)
    parser.add_argument('-thres', '--threshold', help='threshold for trigger',
                        type=float, default=None)
    parser.add_argument('--data_path', help='data path', default=None)
    parser.add_argument('--feature_num', help='data path', type=int,
                        default=None)

    flags = parser.parse_args().__dict__

    for key in flags:
        if flags[key] is not None:
            if not hasattr(config, key):
                print("WARNING: Invalid override with attribute %s" % (key))
            else:
                setattr(config, key, flags[key])

    print(flags)
    runner = Runner(config)

    spec, _ = process_wave('./s_B38A27C4F3E0532_你好的的.wav')
    result = runner.predict(spec)

