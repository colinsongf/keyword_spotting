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
import tensorflow as tf
from utils.common import check_dir, path_join, dense2sparse

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
    window_size = config.step_size
    total_max = y.max()
    total_mean = y.mean()
    threshold_max = total_max / 4
    threshold_mean = (np.percentile(y, 50) + total_mean) / 2
    while start < end:
        if y[start:start + window_size].max() > threshold_max \
                and y[start:start + window_size].mean() > threshold_mean:
            break
        start += window_size
    assert (start < end)
    while end > start:
        if np.percentile(y[end - window_size: end], 99) > threshold_max \
                and y[start:start + window_size].mean() > threshold_mean:
            break
        end -= window_size
    return point2frame(start), point2frame(end)


def dense_to_ont_hot(labels_dense, num_classes):
    len = labels_dense.shape[0]
    labels_one_hot = np.zeros((len, num_classes))
    labels_one_hot[np.arange(len), labels_dense] = 1
    return labels_one_hot


def process_wave(f):
    y, sr = librosa.load(f, sr=config.samplerate)
    length = len(y)
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

    return mel_spectrogram, y


def make_example(f, time_label):
    print(f)
    spectrogram, wave = process_wave(f)

    label = np.zeros(spectrogram.shape[0], dtype=np.int32)
    if len(time_label) > 0:
        for t in time_label:
            word = t[0]
            start_frame, end_frame = adjust(wave, t[1], t[2])
            # print(t[1], t[2])
            label[start_frame:end_frame] = word

    label_indices, label_values, label_len = dense2sparse(label)

    spectrogram = spectrogram.tolist()
    example = tf.train.SequenceExample()
    example.context.feature["label_indices"]. \
        int64_list.value.extend(label_indices)
    example.context.feature["label_values"]. \
        int64_list.value.extend(label_values)
    example.context.feature["label_len"]. \
        int64_list.value.append(label_len)
    fl_audio = example.feature_lists.feature_list["audio"]

    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)

    return example


def make_tfrecord(audio_list, time_list, fname):
    example_list = [make_example(path_join(wave_train_dir, f), time) for f, time in zip(audio_list, time_list)]
    writer = tf.python_io.TFRecordWriter(path_join(config.data_path, fname))
    for ex in example_list:
        writer.write(ex.SerializeToString())
    writer.close()


def process_valid_data(f, fname, correctness):
    print(f)

    mel_spectrogram, y = process_wave(f)

    data = np.stack([mel_spectrogram] * 1)
    # print('data shape is ', data.shape)

    seqLengths = np.asarray([mel_spectrogram.shape[0]] * 1, dtype=np.int32)
    # print(seqLengths.shape)

    return data, seqLengths, fname, correctness


def generate_valid_data(pkl_name):
    with open(pkl_name, 'rb')as f:
        valid_files = pickle.load(f)

    valid_tuples = [process_valid_data(wave_valid_dir + f[0], f[0], f[1][0]) for f in valid_files]
    dump2npy(valid_tuples, save_valid_dir)


def dump2npy(tuples, path):
    # tuple (data,labels,seqLengths)

    maxLen = max([t[0].shape[1] for t in tuples])
    print('max lengths is %d' % maxLen)

    data = np.concatenate(
        [np.pad(t[0], pad_width=((0, 0), (0, maxLen - t[0].shape[1]), (0, 0)), mode='constant', constant_values=0) for t
         in tuples])
    seqLengths = np.concatenate([t[1] for t in tuples])
    print(data.shape)
    print(seqLengths.shape)
    np.save(path_join(path, 'wave.npy'), data)
    np.save(path_join(path, 'seqLen.npy'), seqLengths)
    print('data saved in %s' % path)

    files = [t[2] for t in tuples]
    with open(path_join(path, 'filename.pkl'), 'wb') as f:
        pickle.dump(files, f)

    correctness = [t[3] for t in tuples]
    with open(path_join(path, 'correctness.pkl'), 'wb') as f:
        pickle.dump(correctness, f)


def sort_wave(pkl_path):
    def get_len(f):
        y, sr = librosa.load(f, sr=config.samplerate)
        return len(y)

    with open(pkl_path, "rb") as f:
        training_data = pickle.load(f)
        sorted_data = sorted(training_data, key=lambda a: get_len(wave_train_dir + a[0]))
    with open(pkl_path + '.sorted', "wb") as f:
        pickle.dump(sorted_data, f)


def filter_wave(pkl_path):
    with open(pkl_path, "rb") as f:
        training_data = pickle.load(f)
    print('number before filter:', len(training_data))
    filter_out = []
    for i in training_data:
        if len(i[1]) > 0:
            drop = False
            for t in i[1]:
                if (t[2] - t[1] > 3):
                    drop = True
                    break
            if not drop:
                filter_out.append(i)
        else:
            filter_out.append(i)
    with open(pkl_path + '.filtered', "wb") as f:
        pickle.dump(filter_out, f)
    print('number after filter:', len(filter_out))


if __name__ == '__main__':
    # sort_wave(wave_train_dir + "segment_nihaolele_extra.pkl")
    # filter_wave(wave_train_dir + "segment_nihaolele_extra.pkl.sorted")

    # generate_valid_data(wave_valid_dir + "valid.pkl")
    make_example(wave_train_dir+'azure_228965.wav',[[1, 4.12, 8.88]])

# train_tuples = []
# with open(wave_train_dir + "segment_lele_extra.pkl", "rb") as f:
#     labels = pickle.load(f)
#     print(labels[0])
# train_tuples = [process_record(wave_train_dir + f, f, time_label) for f, time_label in labels]
#
# with open(wave_train_dir + "neg_lele_extra.pkl", 'rb') as f:
#     labels = pickle.load(f)
# train_tuples += [process_record(wave_train_dir + f, f, []) for f, _ in labels]
# dump2npy(train_tuples, save_train_dir, True, False)

# test_data()





# test(wave_train_dir + '160.wav')
#
# a = linear2mel(audio2linear(librosa.load(wave_train_dir + '1.wav', sr=samplerate)[0]))
# b = np.load('mel.npy')
# print(a.max())
# print(a.mean())
# print(b.shape)
