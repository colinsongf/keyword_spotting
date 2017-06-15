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
from utils.common import check_dir, path_join, increment_id

config = get_config()

wave_train_dir = config.rawdata_path + 'train/'
wave_valid_dir = config.rawdata_path + 'valid/'

save_train_dir = path_join(config.data_path, 'train/')
save_valid_dir = path_join(config.data_path, 'valid/')

global_len = []
temp_list = []
error_list = []


def time2frame(second, sr=config.samplerate, n_fft=config.fft_size,
               step_size=config.step_size):
    return int((second * sr - (n_fft // 2)) / step_size) if second > 0 else 0


def point2frame(point, sr=config.samplerate, n_fft=config.fft_size,
                step_size=config.step_size):
    return (point - (n_fft // 2)) // step_size


def time2point(second, sr=config.samplerate):
    return int(second * sr)


def convert_label(frame_label):
    lele_label = []
    whole_label = []
    for i in frame_label:
        if i[0] == 2:
            lele_label.append([1] + i[1:3])
    if len(frame_label) == 2 and frame_label[0][0] == 1:
        whole_label.append([1, frame_label[0][1], frame_label[1][2]])
    return frame_label, lele_label, whole_label


def adjust(y, start, end):
    start = max(time2point(start), 0)
    end = min(len(y), time2point(end))
    y = np.abs(y)
    window_size = config.step_size
    total_max = np.percentile(y, 99)
    total_mean = y.mean()
    threshold_max = total_max / 3.5
    threshold_mean = (np.percentile(y, 50) + total_mean) / 2
    while start < end:
        if y[start:start + window_size].max() > threshold_max:
            break
        start += window_size
    if start >= end:
        return (None, None)
    while end > start:
        if y[end - window_size: end].max() > threshold_max:
            break
        end -= window_size
    if start >= end:
        return (None, None)
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
            librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size,
                                           hop_length=config.step_size,
                                           power=2.,
                                           fmin=300,
                                           fmax=8000,
                                           n_mels=config.num_features))
    elif config.spectrogram == 'mfcc':
        mel_spectrogram = np.transpose(
            librosa.feature.mfcc(y, sr=sr, n_mfcc=config.num_features,
                                 n_fft=config.fft_size,
                                 hop_length=config.step_size,
                                 power=2.,
                                 fmin=300, fmax=8000,
                                 n_mels=config.num_features))
    else:
        raise (Exception('spectrogram %s not defined' % config.spectrogram))
    return mel_spectrogram, y


def make_record(f, time_label):
    # print(f)
    # print(time_label)
    spectrogram, wave = process_wave(f)
    seq_len = spectrogram.shape[0]
    labels = []  # nihaolele   lele  whole
    for i in range(3):
        labels.append(np.zeros(seq_len, dtype=np.int32))
    if len(time_label) > 0:
        new_label = []
        for t in time_label:
            start_frame, end_frame = adjust(wave, t[1], t[2])
            if start_frame is None:
                print('can not process this record')
                print(f)
                return None, None, None
            new_label.append([t[0], start_frame, end_frame])
        frame_labels = convert_label(new_label)
        assert len(frame_labels) == 3
        for i in range(3):
            for t in frame_labels[i]:
                labels[i][t[1]:t[2]] = t[0]

    one_hots = [dense_to_ont_hot(label, num) for label, num in
                zip(labels, [3, 2, 2])]
    return spectrogram, one_hots, seq_len


def make_trainning_example(spectrogram, one_hots, seq_len):
    spectrogram = spectrogram.tolist()
    one_hots = [one_hot.tolist() for one_hot in one_hots]
    ex = tf.train.SequenceExample()

    ex.context.feature["seq_len"].int64_list.value.append(seq_len)

    fl_audio = ex.feature_lists.feature_list["audio"]
    fl_labels = [ex.feature_lists.feature_list[prefix + "_label"] for prefix in
                 config.label_list]
    assert len(fl_labels) == len(one_hots)
    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)
    for fl_label, one_hot in zip(fl_labels, one_hots):
        for frame_label in one_hot:
            fl_label.feature.add().float_list.value.extend(frame_label)

    return ex


def make_valid_example(spectrogram, seq_len, correctness, name):
    spectrogram = spectrogram.tolist()
    ex = tf.train.SequenceExample()

    ex.context.feature["seq_len"].int64_list.value.append(seq_len)
    ex.context.feature['name'].bytes_list.value.append(
        name.encode(encoding="utf-8"))
    ex.context.feature["correctness"].int64_list.value.append(correctness)

    fl_audio = ex.feature_lists.feature_list["audio"]
    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)

    return ex


def process_valid_data(f, fname, correctness):
    # print(f)

    mel_spectrogram, y = process_wave(f)

    data = np.stack([mel_spectrogram] * 1)
    # print('data shape is ', data.shape)

    seqLengths = np.asarray([mel_spectrogram.shape[0]] * 1, dtype=np.int32)
    # print(seqLengths.shape)

    return data, seqLengths, fname, correctness


def generate_valid_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        wav_list = pickle.load(f)
    audio_list = [i[0] for i in wav_list]
    correctness_list = [i[1] for i in wav_list]
    assert len(audio_list) == len(correctness_list)
    tuple_list = []
    counter = 0
    record_count = 0
    for audio_name, correctness in zip(audio_list, correctness_list):
        spec, _ = process_wave(path_join(wave_valid_dir, audio_name))
        seq_len = seq_len = spec.shape[0]
        tuple_list.append((spec, seq_len, correctness, audio_name))
        counter += 1
        if counter == config.tfrecord_size:
            tuple_list = batch_padding_valid(tuple_list)
            fname = 'valid' + increment_id(record_count, 5) + '.tfrecords'
            ex_list = [make_valid_example(spec, seq_len, correctness, name) for
                       spec, seq_len, correctness, name in tuple_list]
            writer = tf.python_io.TFRecordWriter(
                path_join(path_join(config.data_path, 'valid/'), fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            counter = 0
            tuple_list.clear()
            print(fname, 'created')


def batch_padding_trainning(tup_list):
    new_list = []
    max_len = max([t[2] for t in tup_list])

    for t in tup_list:
        assert (len(t[0]) == len(t[1][0]))
        assert (len(t[0]) == t[2])
        paded_wave = np.pad(t[0], pad_width=(
            (0, max_len - t[0].shape[0]), (0, 0)),
                            mode='constant', constant_values=0)
        paded_labels = [
            np.pad(l, pad_width=((0, max_len - l.shape[0]), (0, 0)),
                   mode='constant', constant_values=0) for l in t[1]]
        new_list.append((paded_wave, paded_labels, t[2]))

    return new_list


def batch_padding_valid(tup_list):
    new_list = []
    max_len = max([t[1] for t in tup_list])

    for t in tup_list:
        paded_wave = np.pad(t[0], pad_width=(
            (0, max_len - t[0].shape[0]), (0, 0)),
                            mode='constant', constant_values=0)
        new_list.append((paded_wave, t[1], t[2], t[3]))

    return new_list


def generate_trainning_data(path):
    with open(path, 'rb') as f:
        wav_list = pickle.load(f)
    audio_list = [i[0] for i in wav_list]
    time_list = [i[1] for i in wav_list]
    assert len(audio_list) == len(time_list)
    tuple_list = []
    counter = 0
    record_count = 0
    for i, audio_name in enumerate(audio_list):
        spec, labels, seq_len = make_record(
            path_join(wave_train_dir, audio_name),
            time_list[i])
        if spec is not None:
            counter += 1
            tuple_list.append((spec, labels, seq_len))
        if counter == config.tfrecord_size:
            tuple_list = batch_padding_trainning(tuple_list)
            fname = 'data' + increment_id(record_count, 5) + '.tfrecords'
            ex_list = [make_trainning_example(spec, labels, seq_len) for
                       spec, labels, seq_len in tuple_list]
            writer = tf.python_io.TFRecordWriter(
                path_join(path_join(config.data_path, 'train/'), fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            counter = 0
            tuple_list.clear()
            print(fname, 'created')


def sort_wave(pkl_path):
    def get_len(f):
        y, sr = librosa.load(f, sr=config.samplerate)
        return len(y)

    with open(pkl_path, "rb") as f:
        training_data = pickle.load(f)
        sorted_data = sorted(training_data,
                             key=lambda a: get_len(wave_train_dir + a[0]))
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


def temp(path):
    with open(path, 'rb') as f:
        wav_list = pickle.load(f)
    audio_list = [i[0] for i in wav_list]
    time_list = [i[1] for i in wav_list]
    assert len(audio_list) == len(time_list)
    tuple_list = []
    counter = 0
    for i, audio_name in enumerate(audio_list):
        spec, labels, seq_len = make_record(
            path_join(wave_train_dir, audio_name),
            time_list[i])
        if spec is not None:
            counter += 1
            temp_list.append((audio_name, time_list[i]))
        else:
            print('error')
            error_list.append((audio_name, time_list[i]))
    with open('labelxxx.pkl', 'wb') as f:
        pickle.dump(temp_list, f)
    with open('error.pkl', 'wb') as f:
        pickle.dump(error_list, f)


def shuffle(pkl_path):
    import random
    new_list = []
    batch = 4096
    with open(pkl_path, 'rb') as f:
        wave_list = pickle.load(f)
    total = 0
    for r in wave_list[18 * batch:19 * batch]:
        if len(r[1]) > 0:
            total += 1
    print(total)
    # for i in range(len(wave_list) // batch):
    #     print('batch', i)
    #
    #     temp = wave_list[i * batch:(i + 1) * batch]
    #     # flag = False
    #     # for k in temp:
    #     #     if len(k[1]) > 0:
    #     #         flag = True
    #     #         break
    #     # if not flag:
    #     #     print('fuck', i)
    #
    #     ok = False
    #     again = 0
    #
    #     while not ok and again < 100:
    #         count = 0
    #         random.shuffle(temp)
    #         # print(temp[32:64])
    #         c = 0
    #         for j in range(batch // 32):
    #             subok = False
    #             small_batch = temp[j * 32:(j + 1) * 32]
    #             for record in small_batch:
    #                 if len(record[1]) > 0:
    #                     subok = True
    #                     break
    #             if subok:
    #                 c += 1
    #                 continue
    #             else:
    #                 print(i, 'again')
    #                 again += 1
    #                 break
    #
    #         if c == batch // 32:
    #             ok = True
    #     new_list.extend(temp)
    # print(len(wave_list))
    # new_list.extend(wave_list[len(wave_list) // batch * batch:])
    # print(len(new_list))
    # with open(pkl_path + '.shuffled', "wb") as f:
    #     pickle.dump(new_list, f)


if __name__ == '__main__':
    check_dir(save_train_dir)
    check_dir(save_valid_dir)

    base_pkl = 'label_5x.pkl'
    # sort_wave(wave_train_dir + base_pkl)
    # filter_wave(wave_train_dir + base_pkl + '.sorted')
    # shuffle(wave_train_dir + base_pkl + '.sorted.filtered')
    generate_trainning_data(
        wave_train_dir + base_pkl + '.sorted.filtered.shuffled')

    # generate_valid_data(wave_valid_dir + "valid.pkl")
    # make_example(wave_train_dir+'azure_228965.wav',[[1, 4.12, 8.88]])
