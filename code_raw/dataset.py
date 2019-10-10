# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
import pickle, os


def resample(sig, target_point_num=None):
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def transform(sig, train=False):
    sig = resample(sig, config.target_point_num)
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


def qrs_detect(data_path, df=None):
    if df is None:
        df = pd.read_csv(data_path, sep=' ')
    cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # detect qrs peak
    sep_tol = 100
    grad = np.diff(df[cols].values, axis=0) ** 2
    grad_2 = np.sum(grad, axis=1)
    # grad_2 = np.sum(df[cols].values**2, axis=1)
    def get_peak(tol):
        cands = np.nonzero(grad_2 > tol)[0]
        qrs_peak = []
        if len(cands) > 5:
            cur_group = [cands[0]]
            for i in range(1, len(cands)):
                if cands[i] - cands[i-1] < sep_tol:
                    cur_group.append(cands[i])
                else:
                    qrs_peak.append(int(np.mean(cur_group)))
                    cur_group = [cands[i]]
            if cur_group:
                qrs_peak.append(int(np.mean(cur_group)))
        return qrs_peak
    for peak_tol in [np.max(grad_2) / 3, 10000, 5000, 2000]:
        qrs_peak = get_peak(peak_tol)
        if len(qrs_peak) > 5:
            break
    # get T and start
    n = len(qrs_peak)
    T = (qrs_peak[-1] - qrs_peak[0]) / (n - 1)
    T0 = (n - 1) * n / 2 - np.mean(qrs_peak) / T
    return qrs_peak, T, T0


def load_data_x(file_path, train=True):
    df = pd.read_csv(file_path, sep=' ')
    # position embedding
    # _, T, T0 = qrs_detect(file_path, df)
    # base = np.arange(5000).reshape(-1, 1)
    # pos_sin = np.sin(2 * np.pi * (base / T + T0))
    # pos_cos = np.cos(2 * np.pi * (base / T + T0))
    # data = np.concatenate([df.values / 100, pos_sin, pos_cos], axis=-1)
    data = df.values
    x = transform(data, train)
    return x


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, mode='train'):
        super(ECGDataset, self).__init__()
        self.mode = mode
        self.extra_data = self.load_extra_data()
        self.train = mode in ['train', 'all']
        self.data_dir = config.train_dir if mode != 'test' else config.test_dir
        dd = torch.load(data_path)
        if mode == 'train':
            self.data = dd['train']
        elif mode == 'val':
            self.data = dd['val']
        elif mode == 'all':
            self.data = dd['train'] + dd['val']
        elif mode == 'test':
            self.data = dd['test']
        else:
            raise NameError

        self.idx2name = dd['idx2name']
        # label info
        if mode != 'test':
            self.file2idx = dd['file2idx']
            self.wc = 1. / np.log(dd['wc'])

    def load_extra_data(self):
        data = {}
        for path in config.train_label + [config.test_label]:
            with open(path, encoding='utf-8') as f:
                for line in f.readlines():
                    a = line.rstrip('\n').split('\t')
                    id_ = a[0]
                    age, gender = 0.0, 0.0
                    if a[1]:
                        age = int(a[1])/100
                    if a[2]:
                        gender = 1.0 if a[2] == 'MALE' else -1.0
                    data[id_] = np.array([age, gender])
        return data

    def __getitem__(self, index):
        id_pair = self.data[index]
        id_path, fid = id_pair
        file_path = os.path.join(id_path, fid)
        x = load_data_x(file_path, self.train)
        extra = torch.tensor(self.extra_data[fid], dtype=torch.float32)
        # y
        if self.mode != 'test':
            target = np.zeros(config.num_classes)
            target[self.file2idx[id_pair]] = 1
            target = torch.tensor(target, dtype=torch.float32)
            return x, extra, target
        else:
            return x, extra

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    print(d[0])
