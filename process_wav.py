# encoding: utf-8

'''

@author: ZiqiLiu


@file: process_wav.py

@time: 2017/5/19 下午2:28

@desc:
'''
import librosa
import numpy as np
from config.config import get_config
from glob2 import glob
import os
import pickle
from utils.common import check_dir, path_join

config = get_config()
import matplotlib.pyplot as plt

wave_train_dir = './rawdata/HelloLeLe824/'
wave_neg_train_dir = './rawdata/neg_wav/'
wave_valid_dir = './rawdata/valid/'

save_train_dir = path_join(config.data_path, 'train/')
save_valid_dir = path_join(config.data_path, 'valid/')

check_dir(save_train_dir)
check_dir(save_valid_dir)


def time2frame(second, sr, step_size):
    return int((second * sr - 200) / step_size) if second > 0 else 0


def time2point(second, sr):
    return int(second * sr)


def test(f):
    name = f.split('/')[-1].split('.')[0]
    y, sr = librosa.load(f, sr=config.samplerate)
    print(y.shape)
    seg = y[int(1.76 * 16000): int(2.72 * 16000)]
    begin = y[:int(1.76 * 16000)]
    print((np.abs(begin).mean()))
    print(np.percentile(np.abs(y), 50))
    # print('sr:' + str(sr)))

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size, hop_length=config.step_size, power=2.,
                                       n_mels=config.num_features))
    mfcc = np.transpose(
        librosa.feature.mfcc(y, sr=sr, n_mfcc=40, n_fft=config.fft_size, hop_length=config.step_size, power=2.,
                             n_mels=config.num_features))

    print(mel_spectrogram.shape)
    print(mel_spectrogram.max())
    print(mel_spectrogram.mean())


def audio2linear(audio):
    stft_matrix = librosa.core.stft(
        y=audio,
        n_fft=config.fft_size,
        hop_length=config.step_size,

    )  # shape=(1 + n_fft/2, t)
    linearspec = np.abs(stft_matrix) ** 2
    return linearspec


def linear2mel(linearspec):
    melspec = librosa.feature.melspectrogram(
        S=linearspec,
        sr=config.samplerate,
        n_fft=config.fft_size,
        hop_length=config.step_size,
        power=2.,
        n_mels=20
    )
    melW = librosa.filters.mel(sr=config.samplerate, n_fft=config.fft_size, n_mels=config.num_features)
    # melW /= np.max(melW, axis=-1)[:, None]
    # print(melW.shape)
    melX = np.dot(melW, linearspec)
    return melspec


def dense_to_ont_hot(labels_dense, num_classes):
    len = labels_dense.shape[0]
    labels_one_hot = np.zeros((len, num_classes))
    labels_one_hot[np.arange(len), labels_dense] = 1
    return labels_one_hot


def process_record(f, fname, time, correctness=None, copy=1):
    y, sr = librosa.load(f, sr=config.samplerate)
    # print(len(y))
    # print('sr:' + str(sr))
    # print(fname)

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size, hop_length=config.step_size, power=2., fmin=300,
                                       fmax=8000, n_mels=config.num_features))

    # mfcc = np.transpose(
    #     librosa.feature.mfcc(y, sr=sr, n_mfcc=40, n_fft=config.fft_size, hop_length=config.step_size, power=2.,
    #                          n_mels=config.num_features))

    data = np.stack([mel_spectrogram] * copy)
    # print('data shape is ', data.shape)

    label = np.zeros(mel_spectrogram.shape[0], dtype=np.int32)
    if len(time) > 0:

        for t in time:
            word = t[0]
            start_frame = time2frame(t[1], sr, config.step_size)
            end_frame = time2frame(t[2], sr, config.step_size)
            # print(t[1], t[2])
            label[start_frame:end_frame] = word
    label = dense_to_ont_hot(label, config.num_classes)

    # np.set_printoptions(precision=1, threshold=np.inf, suppress=True)
    # print(label)
    # with open('label.txt', 'a') as out:
    #     out.write(str(label))
    #     out.write('\n-----------------------\n')

    labels = np.stack([label] * copy)
    # print(labels.shape)

    seqLengths = np.asarray([mel_spectrogram.shape[0]] * copy, dtype=np.int32)
    # print(seqLengths.shape)

    return data, labels, seqLengths, fname, correctness


def dump2npy(tuples, path, keep_filename=False, keep_correctness=False):
    # tuple (data,labels,seqLengths)

    maxLen = max([t[0].shape[1] for t in tuples])
    print('max lengths is %d' % maxLen)

    data = np.concatenate(
        [np.pad(t[0], pad_width=((0, 0), (0, maxLen - t[0].shape[1]), (0, 0)), mode='constant', constant_values=0) for t
         in tuples])
    labels = np.concatenate(
        [np.pad(t[1], pad_width=((0, 0), (0, maxLen - t[0].shape[1]), (0, 0)), mode='constant', constant_values=0) for t
         in
         tuples])
    seqLengths = np.concatenate([t[2] for t in tuples])
    print(data.shape)
    print(labels.shape)
    print(seqLengths.shape)
    np.save(path + 'wave.npy', data)
    np.save(path + 'labels.npy', labels)
    np.save(path + 'seqLen.npy', seqLengths)
    if keep_filename:
        files = [t[3] for t in tuples]
        # print(files)
        # print(files)
        with open(path + 'filename.pkl', 'wb') as f:
            pickle.dump(files, f)
    if keep_correctness:
        correctness = [t[4] for t in tuples]
        # print(correctness)
        # print(len(correctness))
        with open(path + 'correctness.pkl', 'wb') as f:
            pickle.dump(correctness, f)


if __name__ == '__main__':
    # train_files = {"1.wav": [[1, 1.645, 2.042], [2, 2.115, 2.506]],
    #                "2.wav": [[1, 1.167, 1.729], [2, 1.742, 2.206]],
    #                "3.wav": [[1, 0.999, 1.462], [2, 1.507, 1.977]],
    #                "s_25E27F1693018046_乐乐你好.wav": [[2, 2.220, 2.778], [1, 2.826, 3.376]],
    #                "s_53BDDB0117F07D72_乐乐你好.wav": [[2, 1.179, 1.743], [1, 1.803, 2.279]],
    #                "s_80A0481C765C1407_给妈妈发一块钱红包.wav": [],
    #                "s_AD78BEA480AC2F9F_图片可爱.wav": [],
    #                "s_CA2C6940CC4F52E1_是这样吗呵呵.wav": [],
    #                "s_6FA9C35E3679E5FE_啦啦啦你好.wav": [[1, 2.372, 2.954]],
    #                "s_49BCE41F1D122F19_乐乐你好.wav": [[2, 0.190, 0.633], [1, 0.639, 1.052]],
    #                "s_50B6C174A1387639_我要看欢乐颂.wav": [],
    #                "s_480B365B7D93C89B_乐乐打开微信.wav": [[2, 1.835, 2.450]],
    #                "s_A5B6C5D2A4DD4BB9_打开QQ音乐.wav": [],
    #                "s_B38A27C4F3E0532_你好的的.wav": [[1, 1.133, 1.790]],
    #                "s_C505F355CE631684_给冯奎打电话.wav": [],
    #                "s_D3C244933788F6A7_你是谁呀.wav": []
    #                }
    # valid_files = {"s_4E1959B1C3387558_你好乐乐.wav": [[1, 1.401, 2.678], [2, 2.765, 3.964]],
    #                "s_9DC4CAC83D317C69_你好乐乐.wav": [[1, 1.455, 1.987], [2, 2.045, 2.540]],
    #                "s_37B725D70B94FCC4_乐乐你好.wav": [[2, 1.307, 1.843], [1, 2.278, 2.660]],
    #                "s_758249C863F7975B_未打开天气.wav": [],
    #                "s_976396C89212A13B_你好乐乐.wav": [[1, 1.072, 2.325], [2, 2.403, 3.305]],
    #                "s_DD8A53F34A912C76_乐乐你他妈会什么.wav": [[2, 1.139, 1.635]],
    #                "s_F30C105D5129CFEB_给孙鹏发微信吃饭.wav": [],
    #                "s_F193089BC92BAFDF_你好你是傻逼吗.wav": [[1, 0.743, 1.253]],
    #                "01.wav": [[1, 1.645, 2.042], [2, 2.115, 2.506]],
    #                "02.wav": [[1, 1.167, 1.729], [2, 1.742, 2.206]],
    #                "03.wav": [[1, 0.999, 1.462], [2, 1.507, 1.977]],
    #                "s_25E27F1693018046_乐乐你好.wav": [[2, 2.220, 2.778], [1, 2.826, 3.376]],
    #                "s_53BDDB0117F07D72_乐乐你好.wav": [[2, 1.179, 1.743], [1, 1.803, 2.279]],
    #                "s_80A0481C765C1407_给妈妈发一块钱红包.wav": [],
    #                "s_AD78BEA480AC2F9F_图片可爱.wav": [],
    #                "s_CA2C6940CC4F52E1_是这样吗呵呵.wav": [],
    #                "s_6FA9C35E3679E5FE_啦啦啦你好.wav": [[1, 2.372, 2.954]],
    #                "s_49BCE41F1D122F19_乐乐你好.wav": [[2, 0.190, 0.633], [1, 0.639, 1.052]],
    #                "s_50B6C174A1387639_我要看欢乐颂.wav": [],
    #                "s_480B365B7D93C89B_乐乐打开微信.wav": [[2, 1.835, 2.450]],
    #                "s_A5B6C5D2A4DD4BB9_打开QQ音乐.wav": [],
    #                "s_B38A27C4F3E0532_你好的的.wav": [[1, 1.133, 1.790]],
    #                "s_C505F355CE631684_给冯奎打电话.wav": [],
    #                "s_D3C244933788F6A7_你是谁呀.wav": [],
    #                "0.wav": [[1, 0.92, 1.32], [2, 1.36, 1.8]],
    #                "1.wav": [[1, 1.04, 1.8], [2, 1.84, 2.44]],
    #                "2.wav": [[1, 1.12, 1.28], [2, 1.32, 1.64]],
    #                "3.wav": [[1, 0.44, 1.0], [2, 1.04, 1.28]],
    #                "4.wav": [[1, 0.8, 1.12], [2, 1.16, 1.68]],
    #                "5.wav": [[1, 1.52, 1.72], [2, 1.76, 2.04]],
    #                "6.wav": [[1, 1.2, 1.4], [2, 1.44, 1.68]],
    #                "7.wav": [[1, 1.76, 1.96], [2, 2.0, 2.24]],
    #                "8.wav": [[1, 1.36, 1.64], [2, 1.68, 2.0]],
    #                "9.wav": [[1, 1.32, 1.48], [2, 1.52, 1.76]],
    #                "10.wav": [[1, 1.08, 1.32], [2, 1.36, 1.68]],
    #                "11.wav": [[1, 1.44, 1.6], [2, 1.64, 1.92]],
    #                "12.wav": [[1, 0.92, 1.16], [2, 1.2, 1.44]],
    #                "13.wav": [[1, 1.4, 1.88], [2, 1.92, 2.4]],
    #                "14.wav": [[1, 1.12, 1.4], [2, 1.44, 1.72]],
    #                "15.wav": [[1, 0.72, 0.96], [2, 1.0, 1.32]],
    #                "16.wav": [[1, 0.72, 0.96], [2, 1.0, 1.32]],
    #                "17.wav": [[1, 0.6, 1.08], [2, 1.12, 1.6]],
    #                "18.wav": [[1, 0.44, 0.92], [2, 0.96, 1.36]],
    #                "19.wav": [[1, 1.24, 1.4], [2, 1.44, 1.76]],
    #                "20.wav": [[1, 0.96, 1.28], [2, 1.32, 1.72]],
    #                "21.wav": [[1, 1.28, 1.52], [2, 1.56, 1.96]],
    #                "22.wav": [[1, 0.96, 1.28], [2, 1.32, 1.64]],
    #                "23.wav": [[1, 1.32, 1.68], [2, 1.72, 2.04]],
    #                "24.wav": [[1, 1.96, 2.24], [2, 2.28, 2.6]],
    #                "25.wav": [[1, 1.12, 1.44], [2, 1.48, 1.88]],
    #                "26.wav": [[1, 1.12, 1.44], [2, 1.48, 1.88]],
    #                "27.wav": [[1, 1.36, 1.88], [2, 1.92, 2.4]],
    #                "28.wav": [[1, 1.64, 2.16], [2, 2.2, 2.68]],
    #                "29.wav": [[1, 1.2, 1.76], [2, 1.8, 2.24]],
    #                "30.wav": [[1, 1.08, 1.64], [2, 1.68, 2.16]],
    #                "31.wav": [[1, 1.2, 1.76], [2, 1.8, 2.28]],
    #                "32.wav": [[1, 1.28, 1.8], [2, 1.84, 2.4]],
    #                "33.wav": [[1, 2.92, 3.24], [2, 3.28, 3.6]],
    #                "34.wav": [[1, 1.2, 1.56], [2, 1.6, 2.04]],
    #                "35.wav": [[1, 2.28, 2.8], [2, 2.84, 3.24]],
    #                "36.wav": [[1, 1.48, 1.88], [2, 1.92, 2.24]],
    #                "37.wav": [[1, 0.8, 1.32], [2, 1.36, 1.76]],
    #                "38.wav": [[1, 1.56, 2.04], [2, 2.08, 2.56]],
    #                "39.wav": [[1, 1.44, 1.92], [2, 1.96, 2.52]]
    #                }
    #
    #
    # train_tuples = [process_file(wave_train_dir + f, f, train_files[f], 1) for f in train_files]
    # dump2npy(train_tuples, save_train_dir, True)
    #

    # with open('./rawdata/valid/valid.pkl', 'rb')as f:
    #     valid_files = pickle.load(f)
    #
    # valid_tuples = [process_record(wave_valid_dir + f, f, valid_files[f][1], valid_files[f][0], 1) for f in valid_files]
    # dump2npy(valid_tuples, save_valid_dir, True, True)
    #
    # train_tuples = []
    #
    # with open(wave_train_dir + "segment.pkl", "rb") as f:
    #     labels = pickle.load(f)
    #     print(labels[0])
    # train_tuples = [process_record(wave_train_dir + f + '.wav', f, time_label) for f, time_label in labels]
    #
    # with open(wave_neg_train_dir + "neg-label-name.pkl", 'rb') as f:
    #     labels = pickle.load(f)
    # train_tuples += [process_record(wave_neg_train_dir + f + '.wav', f, []) for _, f in labels]
    # dump2npy(train_tuples, save_train_dir, True, False)



    test(wave_train_dir + '59.wav')
    #
    # a = linear2mel(audio2linear(librosa.load(wave_train_dir + '1.wav', sr=samplerate)[0]))
    # b = np.load('mel.npy')
    # print(a.max())
    # print(a.mean())
    # print(b.shape)
