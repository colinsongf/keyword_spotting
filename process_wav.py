# encoding: utf-8

'''

@author: ZiqiLiu


@file: process_wav.py

@time: 2017/5/19 下午2:28

@desc:
'''
import librosa
import numpy as np
from config.attention_config import get_config
import pickle
import tensorflow as tf
from utils.common import check_dir, path_join, increment_id

config = get_config()

wave_train_dir = config.rawdata_path + 'train/'
wave_valid_dir = config.rawdata_path + 'valid/'
wave_noise_dir = config.rawdata_path + 'noise/'

save_train_dir = config.train_path
save_valid_dir = config.valid_path
save_noise_dir = config.noise_path

global_len = []
temp_list = []
error_list = []

label_dict = config.label_dict


def pre_emphasis(signal, coefficient=0.97):
    '''对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def time2frame(second, sr=config.samplerate, n_fft=config.fft_size,
               step_size=config.hop_size):
    return int((second * sr - (n_fft // 2)) / step_size) if second > 0 else 0


def point2frame(point, sr=config.samplerate, n_fft=config.fft_size,
                step_size=config.hop_size):
    return (point - (n_fft // 2)) // step_size


def convert_label(label):
    assert len(label) > 0
    label_values = [0]
    for c in label:
        label_values.append(label_dict.get(c, 4))
        label_values.append(0)
    label_shape = len(label_values)
    label_indices = range(len(label_values))

    return label_values, label_indices, label_shape


def process_stft(f):

    y, sr = librosa.load(f, sr=config.samplerate)
    if config.pre_emphasis:
        y = pre_emphasis(y)
    linearspec = np.transpose(np.abs(
        librosa.core.stft(y, config.fft_size,
                          config.hop_size)))

    return linearspec, y


def process_mel(f):
    y, sr = librosa.load(f, sr=config.samplerate)

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size,
                                       hop_length=config.hop_size,
                                       power=2.,
                                       fmin=300,
                                       fmax=8000,
                                       n_mels=config.num_features))

    return mel_spectrogram, y


def make_record(f, label):
    # print(f)
    # print(text)
    spectrogram, wave = process_stft(f)
    seq_len = spectrogram.shape[0]
    label_values, label_indices, label_shape = convert_label(label)

    return spectrogram, seq_len, label_values, label_indices, label_shape


def make_trainning_example(spectrogram, seq_len, label_values, label_indices,
                           label_shape):
    spectrogram = spectrogram.tolist()
    ex = tf.train.SequenceExample()

    ex.context.feature["seq_len"].int64_list.value.append(seq_len)
    ex.context.feature["label_values"].int64_list.value.extend(label_values)
    ex.context.feature["label_indices"].int64_list.value.extend(label_indices)
    ex.context.feature["label_shape"].int64_list.value.append(label_shape)

    fl_audio = ex.feature_lists.feature_list["audio"]

    if label_shape > seq_len:
        raise Exception('invalid label!!!!')

    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)
    return ex


def make_noise_example(spectrogram):
    spectrogram = spectrogram.tolist()
    ex = tf.train.SequenceExample()
    ex.context.feature["seq_len"].int64_list.value.append(
        config.max_sequence_length)
    fl_audio = ex.feature_lists.feature_list["audio"]

    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)

    return ex


def make_valid_example(spectrogram, seq_len, correctness, label, name):
    spectrogram = spectrogram.tolist()
    ex = tf.train.SequenceExample()

    ex.context.feature["seq_len"].int64_list.value.append(seq_len)
    ex.context.feature['label'].int64_list.value.extend(label)
    ex.context.feature['name'].bytes_list.value.append(
        name.encode(encoding="utf-8"))
    ex.context.feature["correctness"].int64_list.value.append(correctness)

    fl_audio = ex.feature_lists.feature_list["audio"]
    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)

    return ex


def batch_padding_trainning(tup_list):
    new_list = []
    max_len = max([t[1] for t in tup_list])

    for t in tup_list:
        assert (len(t[0]) == t[1])
        paded_wave = np.pad(t[0], pad_width=(
            (0, max_len - t[0].shape[0]), (0, 0)),
                            mode='constant', constant_values=0)

        new_list.append((paded_wave, t[1], t[2], t[3], t[4]))
    return new_list


def batch_padding_valid(tup_list):
    new_list = []
    max_len = max([t[1] for t in tup_list])

    for t in tup_list:
        paded_wave = np.pad(t[0], pad_width=(
            (0, max_len - t[0].shape[0]), (0, 0)),
                            mode='constant', constant_values=0)
        new_list.append((paded_wave, t[1], t[2], t[3], t[4]))

    return new_list


def generate_valid_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        wav_list = pickle.load(f)
    print('read pkl from %s' % f)
    audio_list = [i[0] for i in wav_list]
    correctness_list = [i[1] for i in wav_list]
    label_list = [i[2] for i in wav_list]
    assert len(audio_list) == len(correctness_list)
    tuple_list = []
    counter = 0
    record_count = 0
    for audio_name, correctness, label in zip(audio_list, correctness_list,
                                              label_list):
        spectrogram, wave = process_stft(path_join(wave_valid_dir, audio_name))
        seq_len = spectrogram.shape[0]
        label_values, _, _ = convert_label(label)

        tuple_list.append(
            (spectrogram, seq_len, correctness, label_values, audio_name))
        counter += 1
        if counter == config.tfrecord_size:
            tuple_list = batch_padding_valid(tuple_list)
            fname = 'valid' + increment_id(record_count, 5) + '.tfrecords'
            ex_list = [
                make_valid_example(spec, seq_len, correctness, label_values,
                                   audio_name) for
                spec, seq_len, correctness, label_values, audio_name in
                tuple_list]
            writer = tf.python_io.TFRecordWriter(
                path_join(save_valid_dir, fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            counter = 0
            tuple_list.clear()
            print(fname, 'created')
    print('save in %s' % save_valid_dir)


def generate_trainning_data(path):
    with open(path, 'rb') as f:
        wav_list = pickle.load(f)
    print('read pkl from %s' % f)
    audio_list = [i[0] for i in wav_list]
    label_list = [i[1] for i in wav_list]
    text_list = [i[2] for i in wav_list]
    assert len(audio_list) == len(text_list)
    tuple_list = []
    counter = 0
    record_count = 0
    for i, audio_name in enumerate(audio_list):
        spec, seq_len, label_values, label_indices, label_shape = make_record(
            path_join(wave_train_dir, audio_name),
            label_list[i])
        # print(text_list[i])
        # print(label_values)
        if spec is not None:
            counter += 1
            tuple_list.append(
                (spec, seq_len, label_values, label_indices, label_shape))
        if counter == config.tfrecord_size:
            tuple_list = batch_padding_trainning(tuple_list)
            fname = 'data' + increment_id(record_count, 5) + '.tfrecords'
            ex_list = [make_trainning_example(spec, seq_len, label_values,
                                              label_indices, label_shape) for
                       spec, seq_len, label_values, label_indices, label_shape
                       in tuple_list]
            writer = tf.python_io.TFRecordWriter(
                path_join(save_train_dir, fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            counter = 0
            tuple_list.clear()
            print(fname, 'created')
    print('save in %s' % save_train_dir)


def generate_noise_data(path):
    with open(path, 'rb') as f:
        audio_list = pickle.load(f)
        print('read pkl from ', f)
    spec_list = []
    counter = 0
    record_count = 0
    for i, audio_name in enumerate(audio_list):
        spec, y = process_stft(path_join(wave_noise_dir, audio_name))
        spec_list.append(spec)
        counter += 1
        if counter == config.tfrecord_size:
            spec_list = [
                expand_spectrogram(s, config.max_sequence_length) for s in
                spec_list]

            fname = 'noise' + increment_id(record_count, 5) + '.tfrecords'
            ex_list = [make_noise_example(spec) for spec in spec_list]
            writer = tf.python_io.TFRecordWriter(
                path_join(save_noise_dir, fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            counter = 0
            spec_list.clear()
            print(fname, 'created')
    print('save in %s' % save_noise_dir)


def expand_spectrogram(spec, target_len):
    times = target_len // spec.shape[0]
    expand_spec = spec
    for i in range(times):
        expand_spec = np.concatenate((expand_spec, spec), 0)
    return expand_spec[:target_len]


def sort_wave(pkl_path):
    def get_len(f):
        y, sr = librosa.load(f, sr=config.samplerate)
        return len(y)

    import re
    dir = re.sub(r'[^//]+.pkl', '', pkl_path)

    with open(pkl_path, "rb") as f:
        training_data = pickle.load(f)
        sorted_data = sorted(training_data,
                             key=lambda a: get_len(dir + a[0]))
    with open(pkl_path + '.sorted', "wb") as f:
        pickle.dump(sorted_data, f)

    y, sr = librosa.load(dir + sorted_data[0][0])
    print(len(y))


def shuffle(pkl_path):
    import random
    new_list = []
    batch = 4096
    with open(pkl_path, 'rb') as f:
        wave_list = pickle.load(f)
    total = 0

    for i in range(len(wave_list) // batch):
        print('batch', i)

        temp = wave_list[i * batch:(i + 1) * batch]
        # flag = False
        # for k in temp:
        #     if len(k[1]) > 0:
        #         flag = True
        #         break
        # if not flag:
        #     print('fuck', i)

        ok = False
        again = 0

        while not ok and again < 100:
            count = 0
            random.shuffle(temp)
            c = 0
            for j in range(batch // 32):
                subok = False
                small_batch = temp[j * 32:(j + 1) * 32]
                for record in small_batch:
                    if '你好' in record[2] or '乐乐' in record[2]:
                        subok = True
                        break
                if subok:
                    c += 1
                    continue
                else:
                    print(i, 'again')
                    again += 1
                    break

            if c == batch // 32:
                ok = True
        new_list.extend(temp)
    print(len(wave_list))
    new_list.extend(wave_list[len(wave_list) // batch * batch:])
    print(len(new_list))
    with open(pkl_path + '.shuffled', "wb") as f:
        pickle.dump(new_list, f)


if __name__ == '__main__':
    check_dir(save_train_dir)
    check_dir(save_valid_dir)
    check_dir(save_noise_dir)

    base_pkl = 'ctc_23w.pkl'
    # sort_wave(wave_train_dir + base_pkl)
    # # shuffle(wave_train_dir + base_pkl + '.sorted')
    # generate_trainning_data(
    #     wave_train_dir + base_pkl + '.sorted.shuffled')

    # sort_wave(wave_valid_dir + "ctc_valid_pinyin.pkl")
    generate_valid_data(wave_valid_dir + "ctc_valid.pkl.sorted")

    # generate_noise_data(wave_noise_dir + 'noise.pkl')
