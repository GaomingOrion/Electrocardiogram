# -*- coding: utf-8 -*-
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from torch import nn
from numba import jit


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def offline_test(sub, ans):
    file1 = open(sub, encoding='utf-8').read().rstrip('\n').split('\n')
    file2 = open(ans, encoding='utf-8').read().rstrip('\n').split('\n')
    sub1, ans1, all1 = 0, 0, 0
    for i in range(len(file1)):
        a = set(file1[i].rstrip().split('\t')[3:])
        b = set(file2[i].rstrip().split('\t')[3:])
        sub1 += len(a)
        ans1 += len(b)
        all1 += len(a & b)
    p = all1 / sub1
    r = all1 / ans1
    f1 = 2 * all1 / (sub1 + ans1)
    print('precision:%.4f recall:%.4f f1:%.4f' % (p, r, f1))


if __name__ == '__main__':
    ans = '../data/hefei_round1_ansA_20191008.txt'
    offline_test('../submit/subA_201910020050_827.txt', ans)
    sub = '../submit/subA_1007.txt'
    offline_test(sub, ans)