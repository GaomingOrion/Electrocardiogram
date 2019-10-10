# -*- coding: utf-8 -*-
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from torch import nn
from numba import jit
from config import config


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


def get_groups():
    arrys = open(config.arrythmia, encoding='utf-8').read().rstrip().split()
    groups = {'窦性': ['窦性心动过缓', '窦性心动过速', '窦性心律'],
             'ECG': ['正常ECG', '临界ECG', '异常ECG'],
             '电压': ['左心室肥大', '左心室高电压', 'QRS低电压'],
             '心室率': ['快心室率', '慢心室率'],
             'ST': ['T波改变', 'ST段改变', 'ST-T改变'],
             '非特异性ST': ['非特异性T波异常', '非特异性ST段异常', '非特异性ST段与T波异常'],
             '电轴': ['电轴左偏', '电轴右偏'],
             '右束支': ['右束支传导阻滞', '完全性右束支传导阻滞', '不完全性右束支传导阻滞'],
             '左束支': ['左束支传导阻滞', '完全性左束支传导阻滞', '左前分支传导阻滞']}
    groups_set = set()
    for group in groups.values():
        for x in group:
            groups_set.add(x)
    solo = [x for x in arrys if x not in groups_set and x != '双分支传导阻滞']
    return {'multi': groups, 'solo': solo}


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


class WeightedMultilabel(nn.Module):
    def __init__(self, groups, count, device):
        super(WeightedMultilabel, self).__init__()
        self.keys = list(sorted(groups['multi'].keys()))
        self.output_index = [0]
        for k in self.keys:
            self.output_index.append(self.output_index[-1] + len(groups['multi'][k]) + 1)

        self.multi_loss = {}
        for k in groups['multi']:
            # weight = 1 / np.log1p(np.array(count['multi'][k] + [config.num_samples - sum(count['multi'][k])]))
            # weight = torch.tensor(weight / np.sum(weight), dtype=torch.float).to(device)
            self.multi_loss[k] = nn.CrossEntropyLoss()

        solo_weight = 1 / np.log1p(np.array(count['solo']))
        solo_weight = torch.tensor(solo_weight / np.sum(solo_weight), dtype=torch.float).to(device)
        self.solo_loss = nn.BCEWithLogitsLoss(weight=solo_weight)

        self.weights = [np.sum(count['multi'][k]) for k in self.keys]
        self.weights.append(np.mean(count['solo']))
        self.weights = 1 / np.log1p(self.weights)
        self.weights = torch.tensor(self.weights / np.sum(self.weights), dtype=torch.float).to(device)

    def forward(self, outputs, targets_long, targets_float):
        loss = [self.multi_loss[k](outputs[:, self.output_index[i]:self.output_index[i+1]], targets_long[:, i]).unsqueeze(0)
                for i, k in enumerate(self.keys)]
        loss.append(self.solo_loss(outputs[:, self.output_index[-1]:], targets_float).unsqueeze(0))
        loss = torch.cat(loss)
        return (loss * self.weights).sum()


if __name__ == '__main__':
    groups = get_groups()
    dd = torch.load(config.train_data)
    count = dd['count']
    device = torch.device('cuda')
    c = WeightedMultilabel(groups, count, device)

    outputs = torch.randn(64, config.num_classes).to(device)
    targets_long = torch.tensor(np.array([np.argmax(np.random.random([64, len(groups['multi'][k])+1]), axis=1)
                                          for k in sorted(groups['multi'].keys())]).T, dtype=torch.long).to(device)
    targets_float = torch.tensor(np.where(np.random.random([64, len(groups['solo'])]) > 0.5, 1, 0), dtype=torch.float).to(device)
    loss = c(outputs, targets_long, targets_float)