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
import pickle

config = get_config()
import matplotlib.pyplot as plt

fft_size = 400
step_size = 160
samplerate = 16000
wave_train_dir = '../testdata/train/'
wave_valid_dir = '../testdata/valid/'
save_train_dir = '../data/train/'
save_valid_dir = '../data/valid/'


def time2frame(second, sr, step_size):
    return int(second * sr / step_size) if second > 0 else 0


def test(f, begin, end):
    name = f.split('/')[-1].split('.')[0]
    y, sr = librosa.load(f, sr=samplerate)
    # print(len(y))
    # print('sr:' + str(sr))

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=fft_size, hop_length=step_size, power=2., fmin=300,
                                       fmax=8000, n_mels=20))
    print(len(mel_spectrogram))

    start_frame = time2frame(begin, sr, step_size)
    end_frame = time2frame(end, sr, step_size)

    label = np.zeros(mel_spectrogram.shape[0], dtype=np.int32)

    # print(start_frame, end_frame)
    label[start_frame:end_frame] = 1
    np.set_printoptions(precision=8, threshold=np.inf, suppress=True)

    print(mel_spectrogram.shape)
    print(mel_spectrogram)
    with open('mel.txt', 'a') as out:
        out.write(str(mel_spectrogram))
        out.write('\n-----------------------\n')
        # print(labels.shape)


def dense_to_ont_hot(labels_dense, num_classes):
    len = labels_dense.shape[0]
    labels_one_hot = np.zeros((len, num_classes))
    labels_one_hot[np.arange(len), labels_dense] = 1
    return labels_one_hot


def process_file(f, fname, begin, end, copy=1):
    name = f.split('/')[-1].split('.')[0]
    y, sr = librosa.load(f, sr=samplerate)
    # print(len(y))
    # print('sr:' + str(sr))

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=fft_size, hop_length=step_size, power=2., fmin=300,
                                       fmax=8000, n_mels=20))
    print(len(mel_spectrogram))
    data = np.stack([mel_spectrogram] * copy)
    print('data shape is ', data.shape)

    start_frame = time2frame(begin, sr, step_size)
    end_frame = time2frame(end, sr, step_size)

    label = np.zeros(mel_spectrogram.shape[0], dtype=np.int32)

    # print(start_frame, end_frame)
    label[start_frame:end_frame] = 1
    label = dense_to_ont_hot(label, config.num_classes)
    # print(label.shape)
    with open('label.txt', 'a') as out:
        out.write(str(label))
        out.write('\n-----------------------\n')
    labels = np.stack([label] * copy)
    # print(labels.shape)

    seqLengths = np.asarray([mel_spectrogram.shape[0]] * copy, dtype=np.int32)
    # print(seqLengths.shape)

    return data, labels, seqLengths, fname


def dump2npy(tuples, path, keep_filename=False):
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
        print(files)
        # print(files)
        with open(path + 'filename.pkl', 'wb') as f:
            pickle.dump(files, f)


if __name__ == '__main__':
    train_files = {'1.wav': (1.671, 2.5), '2.wav': (1.22, 2.22), '3.wav': (1.0018, 1.952),
                   's_80A0481C765C1407_给妈妈发一块钱红包.wav': (0, 0),
                   's_AD78BEA480AC2F9F_图片可爱.wav': (0, 0),
                   's_CA2C6940CC4F52E1_是这样吗呵呵.wav': (0, 0)}
    valid_files = {'s_4E1959B1C3387558_你好乐乐.wav': (0, 0),
                   's_9DC4CAC83D317C69_你好乐乐.wav': (1.455, 2.517),
                   's_37B725D70B94FCC4_乐乐你好.wav': (0, 0),
                   's_758249C863F7975B_未打开天气.wav': (0, 0),
                   's_976396C89212A13B_你好乐乐.wav': (0, 0),
                   's_DD8A53F34A912C76_乐乐你他妈会什么.wav': (0, 0),
                   's_F30C105D5129CFEB_给孙鹏发微信吃饭.wav': (0, 0),
                   's_F193089BC92BAFDF_你好你是傻逼吗.wav': (0, 0)}

    # valid_files = glob(wave_valid_dir + '*.wav')


    train_tuples = [process_file(wave_train_dir + f, f, train_files[f][0], train_files[f][1], 1) for f in train_files]
    dump2npy(train_tuples, save_train_dir, True)

    valid_tuples = [process_file(wave_valid_dir + f, f, valid_files[f][0], valid_files[f][1], 1) for f in valid_files]
    dump2npy(valid_tuples, save_valid_dir, True)
    # test(wave_train_dir+'1.wav', 1.671, 2.5)
