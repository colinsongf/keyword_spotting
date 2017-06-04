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

wave_train_dir = config.rawdata_path + 'train/'
wave_valid_dir = config.rawdata_path + 'valid/'

save_train_dir = path_join(config.data_path, 'train/')
save_valid_dir = path_join(config.data_path, 'valid/')

check_dir(save_train_dir)
check_dir(save_valid_dir)


def time2frame(second, sr=config.samplerate, n_fft=config.fft_size, step_size=config.step_size):
    return int((second * sr - (n_fft // 2)) / step_size) if second > 0 else 0


def point2frame(point, sr=config.samplerate, n_fft=config.fft_size, step_size=config.step_size):
    return (point - (n_fft // 2)) // step_size


def time2point(second, sr=config.samplerate):
    return int(second * sr)


def adjust(y, start, end):
    start = max(time2point(start), 0)
    end = min(len(y), time2point(end))
    y = np.abs(y)
    window_size = 160
    mean = y.mean()
    quatile = np.percentile(y, 25)
    threshold = (quatile + mean) / 2
    while start < end:
        if (y[start:start + window_size].mean() > threshold):
            break
        start += window_size
    assert (start < end)
    while end > start:
        if (y[end - window_size: end].mean() > threshold):
            break
        end -= window_size
    return point2frame(start), point2frame(end)


def test(f):
    name = f.split('/')[-1].split('.')[0]
    y, sr = librosa.load(f, sr=config.samplerate)
    print(y.shape)
    seg = y[int(1.92 * 16000): int(2.72 * 16000)]
    begin = y[:int(1.76 * 16000)]
    print((np.abs(seg).mean()))
    print(np.percentile(np.abs(y), 50))
    # print('sr:' + str(sr)))
    t = adjust(y, 1.84, 2.84)
    print('======', t)

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size, hop_length=config.step_size, power=2.,
                                       n_mels=config.num_features))
    # mfcc = np.transpose(
    #     librosa.feature.mfcc(y, sr=sr, n_mfcc=40, n_fft=config.fft_size, hop_length=config.step_size, power=2.,
    #                          n_mels=config.num_features))

    print(mel_spectrogram.shape)
    print(mel_spectrogram.max())
    print(mel_spectrogram.mean())


def dense_to_ont_hot(labels_dense, num_classes):
    len = labels_dense.shape[0]
    labels_one_hot = np.zeros((len, num_classes))
    labels_one_hot[np.arange(len), labels_dense] = 1
    return labels_one_hot


def process_record(f, fname, time, correctness=None, copy=1):
    print(f)

    y, sr = librosa.load(f, sr=config.samplerate)

    if config.spectrogram == 'mel':
        mel_spectrogram = np.transpose(
            librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size, hop_length=config.step_size, power=2.,
                                           fmin=300,
                                           fmax=8000, n_mels=config.num_features))
    elif config.spectrogram == 'mfcc':
        mel_spectrogram = np.transpose(
            librosa.feature.mfcc(y, sr=sr, n_mfcc=config.num_features, n_fft=config.fft_size,
                                 hop_length=config.step_size,
                                 power=2.,
                                 fmin=300, fmax=8000, n_mels=config.num_features))
    else:
        raise (Exception('spectrogram %s not defined') % config.spectrogram)

    data = np.stack([mel_spectrogram] * copy)
    # print('data shape is ', data.shape)

    label = np.zeros(mel_spectrogram.shape[0], dtype=np.int32)
    if len(time) > 0:
        for t in time:
            word = t[0]
            start_frame, end_frame = adjust(y, t[1], t[2])
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
    np.save(path_join(path, 'wave.npy'), data)
    np.save(path_join(path, 'labels.npy'), labels)
    np.save(path_join(path, 'seqLen.npy'), seqLengths)
    print('data saved in %s' % path)
    if keep_filename:
        files = [t[3] for t in tuples]
        # print(files)
        # print(files)
        with open(path_join(path, 'filename.pkl'), 'wb') as f:
            pickle.dump(files, f)
    if keep_correctness:
        correctness = [t[4] for t in tuples]
        # print(correctness)
        # print(len(correctness))
        with open(path_join(path, 'correctness.pkl'), 'wb') as f:
            pickle.dump(correctness, f)


def test_data():
    from glob import glob

    pkl = glob('./test/*.pkl')
    for p in pkl:
        with open(p, 'rb')as f:
            valid_files = pickle.load(f)
            path = p.split('.pkl')[0]

        valid_tuples = [process_record(path_join(path, f[0]), f[0], f[1][1], f[1][0], 1) for f in valid_files]
        dump2npy(valid_tuples, path, True, True)


if __name__ == '__main__':
    with open('./rawdata/valid/valid_lele.pkl', 'rb')as f:
        valid_files = pickle.load(f)

    valid_tuples = [process_record(wave_valid_dir + f[0], f[0], f[1][1], f[1][0], 1) for f in valid_files]
    dump2npy(valid_tuples, save_valid_dir, True, True)

    train_tuples = []
    with open(wave_train_dir + "segment_lele.pkl", "rb") as f:
        labels = pickle.load(f)
        print(labels[0])
    train_tuples = [process_record(wave_train_dir + f, f, time_label) for f, time_label in labels]

    with open(wave_train_dir + "neg_lele.pkl", 'rb') as f:
        labels = pickle.load(f)
    train_tuples += [process_record(wave_train_dir + f, f, []) for f, _ in labels]
    dump2npy(train_tuples, save_train_dir, True, False)

    # test_data()





    # test(wave_train_dir + '160.wav')
    #
    # a = linear2mel(audio2linear(librosa.load(wave_train_dir + '1.wav', sr=samplerate)[0]))
    # b = np.load('mel.npy')
    # print(a.max())
    # print(a.mean())
    # print(b.shape)
