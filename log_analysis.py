# encoding: utf-8

'''

@author: ZiqiLiu


@file: log_analysis.py

@time: 2017/6/19 下午4:33

@desc:
'''

import matplotlib.pyplot as plt

log_dir = './logs/'
log_file = 'keyword-adam-2e4-20170619-2.current.txt'
log_file2 = 'keyword-adam-5e3-20170619-3.current_nan.txt'


def read_file(f):
    with open(log_dir + f, 'r') as f:
        logs = [line for line in f]
    return logs


def plot_miss(logs, color, name, limit=-1):
    miss = []
    for line in logs:
        if line.startswith('miss'):
            miss.append(float(line.split(':')[-1]))
    miss = miss[:limit]
    x = range(len(miss))
    print(name, min(miss))
    plt.plot(x, miss, color, linewidth=1, label=name)
    plt.plot(x, [0.2] * len(x), color='black', linewidth=1)
    plt.plot(x, [0.1] * len(x), color='black', linewidth=1)
    plt.xlabel("mini_epoch")  # X轴标签
    plt.ylabel("miss rate")  # Y轴标签


def plot_wer(logs, color, name, limit=-1):
    wer = []
    for line in logs:
        if line.startswith('wer'):
            wer.append(float(line.split()[1]))
    wer = wer[:limit]
    x = range(len(wer))
    print(name, min(wer))
    plt.plot(x, wer, color, linewidth=1, label=name)
    plt.plot(x, [0.2] * len(x), color='black', linewidth=1)
    plt.plot(x, [0.1] * len(x), color='black', linewidth=1)
    plt.xlabel("mini_epoch")  # X轴标签
    plt.ylabel("wer")  # Y轴标签


def plot_lr(logs, color, name, limit=-1):
    lr = []
    step = []
    for line in logs:
        if line.startswith('learning rate'):
            # print(line)
            item = line.split(':')[1].split('global step')
            # print(item)
            lr.append(float(item[0].strip()))
            step.append(int(item[1].strip()))
    lr = lr[:limit]
    step = step[:limit]
    plt.plot(step, lr, color, linewidth=1, label=name)
    plt.xlabel("step")  # X轴标签
    plt.ylabel("lr")  # Y轴标签


def plot_loss(logs, color, name,limit):
    accu_loss = []
    for line in logs:
        if line.startswith('accumulated loss'):
            accu_loss.append(float(line.split('accumulated loss')[-1]))
    accu_loss=accu_loss[:limit]
    x = range(len(accu_loss))

    plt.plot(x, accu_loss, color, linewidth=1, label=name)
    plt.xlabel("epoch")  # X轴标签
    plt.ylabel("total loss")  # Y轴标签


def plot_many(files, plot_fn, limit=-1):
    logs = [read_file(f) for f in files]
    colors = ['r', 'b', 'g', 'm', 'y', 'k']
    plt.figure(figsize=(10, 4))  # 创建绘图对象
    for log, c, f in zip(logs, colors, files):
        plot_fn(log, c, f, limit)
    plt.legend(loc='upper right')
    # plt.savefig('./plot/' + log_file + '.png')
    plt.show()  # 显示图


plot_many([
    'keyword-20170720-rnn-15e3-decay2w-ln-res-keep06.current.txt',
    'keyword-20170722-rnn-15e3-decay2w-ln-res-keep06.current.txt',
    'keyword-20170723-rnn-15e3-decay2w-ln-res-mel40-new.current.txt',
    # 'keyword-5e3-20170630-bgnoise-3l.current.txt',
    # 'keyword-5e3-20170630-bgnoise.current.txt',
    # 'keyword-5e3-20170630-dropout-l3.current.txt',
    # 'keyword-5e3-20170701-dropout-l3-noise-normalized.current.txt',
    # 'keyword-5e3-20170630-dropout-normalize.current.txt',
    # 'keyword-5e3-20170630-dropout2.current.txt',
    # 'keyword-5e3-20170630-whitenoise.current.txt'
],
    plot_miss, )
