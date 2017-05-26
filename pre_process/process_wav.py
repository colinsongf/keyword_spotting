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
wave_train_dir = '../rawdata/HelloLeLe824/'
wave_neg_train_dir = '../rawdata/neg_wav/'
wave_valid_dir = '../rawdata/valid/'
save_train_dir = '../data/train/'
save_valid_dir = '../data/valid/'


# 1:你 2：好 3：乐乐


def time2frame(second, sr, step_size):
    return int((second * sr - 200) / step_size) if second > 0 else 0


def test(f):
    name = f.split('/')[-1].split('.')[0]
    y, sr = librosa.load(f, sr=samplerate)
    # print(len(y))
    # print('sr:' + str(sr))

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=fft_size, hop_length=step_size, power=2.,
                                       n_mels=20))
    # print(mel_spectrogram)
    print(mel_spectrogram.max())
    print(mel_spectrogram.mean())


def audio2linear(audio):
    stft_matrix = librosa.core.stft(
        y=audio,
        n_fft=fft_size,
        hop_length=step_size,

    )  # shape=(1 + n_fft/2, t)
    linearspec = np.abs(stft_matrix) ** 2
    return linearspec


def linear2mel(linearspec):
    melspec = librosa.feature.melspectrogram(
        S=linearspec,
        sr=samplerate,
        n_fft=fft_size,
        hop_length=step_size,
        power=2.,
        n_mels=20
    )
    melW = librosa.filters.mel(sr=samplerate, n_fft=fft_size, n_mels=20)
    # melW /= np.max(melW, axis=-1)[:, None]
    # print(melW.shape)
    melX = np.dot(melW, linearspec)
    return melspec


def dense_to_ont_hot(labels_dense, num_classes):
    len = labels_dense.shape[0]
    labels_one_hot = np.zeros((len, num_classes))
    labels_one_hot[np.arange(len), labels_dense] = 1
    return labels_one_hot


def process_record(f, fname, time, copy=1):
    name = f.split('/')[-1].split('.')[0]
    y, sr = librosa.load(f, sr=samplerate)
    # print(len(y))
    # print('sr:' + str(sr))
    print(fname)

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=fft_size, hop_length=step_size, power=2., fmin=300,
                                       fmax=8000, n_mels=20))

    data = np.stack([mel_spectrogram] * copy)
    # print('data shape is ', data.shape)

    label = np.zeros(mel_spectrogram.shape[0], dtype=np.int32)
    if len(time) > 0:
        for i, t in enumerate(time):
            word = i + 1
            start_frame = time2frame(t[0], sr, step_size)
            end_frame = time2frame(t[0], sr, step_size)
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
    valid_files = {"s_4E1959B1C3387558_你好乐乐.wav": [[1, 1.401, 2.678], [2, 2.765, 3.964]],
                   "s_9DC4CAC83D317C69_你好乐乐.wav": [[1, 1.455, 1.987], [2, 2.045, 2.540]],
                   "s_37B725D70B94FCC4_乐乐你好.wav": [[2, 1.307, 1.843], [1, 2.278, 2.660]],
                   "s_758249C863F7975B_未打开天气.wav": [],
                   "s_976396C89212A13B_你好乐乐.wav": [[1, 1.072, 2.325], [2, 2.403, 3.305]],
                   "s_DD8A53F34A912C76_乐乐你他妈会什么.wav": [[2, 1.139, 1.635]],
                   "s_F30C105D5129CFEB_给孙鹏发微信吃饭.wav": [],
                   "s_F193089BC92BAFDF_你好你是傻逼吗.wav": [[1, 0.743, 1.253]],
                   "01.wav": [[1, 1.645, 2.042], [2, 2.115, 2.506]],
                   "02.wav": [[1, 1.167, 1.729], [2, 1.742, 2.206]],
                   "03.wav": [[1, 0.999, 1.462], [2, 1.507, 1.977]],
                   "s_25E27F1693018046_乐乐你好.wav": [[2, 2.220, 2.778], [1, 2.826, 3.376]],
                   "s_53BDDB0117F07D72_乐乐你好.wav": [[2, 1.179, 1.743], [1, 1.803, 2.279]],
                   "s_80A0481C765C1407_给妈妈发一块钱红包.wav": [],
                   "s_AD78BEA480AC2F9F_图片可爱.wav": [],
                   "s_CA2C6940CC4F52E1_是这样吗呵呵.wav": [],
                   "s_6FA9C35E3679E5FE_啦啦啦你好.wav": [[1, 2.372, 2.954]],
                   "s_49BCE41F1D122F19_乐乐你好.wav": [[2, 0.190, 0.633], [1, 0.639, 1.052]],
                   "s_50B6C174A1387639_我要看欢乐颂.wav": [],
                   "s_480B365B7D93C89B_乐乐打开微信.wav": [[2, 1.835, 2.450]],
                   "s_A5B6C5D2A4DD4BB9_打开QQ音乐.wav": [],
                   "s_B38A27C4F3E0532_你好的的.wav": [[1, 1.133, 1.790]],
                   "s_C505F355CE631684_给冯奎打电话.wav": [],
                   "s_D3C244933788F6A7_你是谁呀.wav": [],
                   "0.wav": [[1, 0, 0], [2, (0.92, 1.32), (1.36, 1.8)]],
                   "1.wav": [[1, 1, 1], [2, (1.04, 1.8), (1.84, 2.44)]],
                   "2.wav": [[1, 2, 2], [2, (1.12, 1.28), (1.32, 1.64)]],
                   "3.wav": [[1, 3, 3], [2, (0.44, 1.0), (1.04, 1.28)]],
                   "4.wav": [[1, 4, 4], [2, (0.8, 1.12), (1.16, 1.68)]],
                   "5.wav": [[1, 5, 5], [2, (1.52, 1.72), (1.76, 2.04)]],
                   "6.wav": [[1, 6, 6], [2, (1.2, 1.4), (1.44, 1.68)]],
                   "7.wav": [[1, 7, 7], [2, (1.76, 1.96), (2.0, 2.24)]]

                   }
    #
    #
    # train_tuples = [process_file(wave_train_dir + f, f, train_files[f], 1) for f in train_files]
    # dump2npy(train_tuples, save_train_dir, True)
    #
    valid_tuples = [process_record(wave_valid_dir + f, f, valid_files[f], 1) for f in valid_files]
    dump2npy(valid_tuples, save_valid_dir, True)
    # test(wave_train_dir + '1.wav')
    #
    # a = linear2mel(audio2linear(librosa.load(wave_train_dir + '1.wav', sr=samplerate)[0]))
    # b = np.load('mel.npy')
    # print(a.max())
    # print(a.mean())
    # print(b.shape)

    with open(wave_train_dir + "segment.pkl", "rb") as f:
        labels = pickle.load(f)
    train_tuples = [process_record(wave_train_dir + f + '.wav', f, time_label) for f, time_label in labels]

    with open(wave_neg_train_dir + "neg-label-name.pkl", 'rb') as f:
        labels = pickle.load(f)
    train_tuples += [process_record(wave_neg_train_dir + f + '.wav', f, []) for _, f in labels]
    dump2npy(train_tuples, save_train_dir, True)
