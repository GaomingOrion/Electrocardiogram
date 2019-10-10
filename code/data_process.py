# -*- coding: utf-8 -*-
import os, torch
import numpy as np
import pandas as pd
from config import config
import utils

# seed
np.random.seed(41)


def name2index():
    arrys = open(config.arrythmia, encoding='utf-8').read().rstrip().split()
    name2idx = {name: i for i, name in enumerate(arrys)}
    return name2idx


def split_data(file2idx, val_ratio=0.1):
    p = config.num_classes
    class_idx = [[] for _ in range(p)]
    all_index = list(file2idx)
    np.random.shuffle(all_index)
    for idx in all_index:
        for i in file2idx[idx]:
            class_idx[i].append(idx)
    class_idx = np.array(class_idx)
    val_index = set()
    for i in range(p-1, 0, -1):
        num = len(class_idx[i])
        val_index |= set(class_idx[i][:int(num*val_ratio)])
    num0 = sum(1 for idx in val_index if file2idx[idx][0] == 0)
    rest_num = int(len(class_idx[0])*val_ratio) - num0
    if rest_num > 0:
        cnt = 0
        for idx in class_idx[0]:
            if idx not in val_index:
                cnt += 1
                val_index.add(idx)
                if cnt == rest_num:
                    break
    train_index = [idx for idx in all_index if idx not in val_index]
    val_index = list(val_index)
    return train_index, val_index


def file2label(path_lst, name2idx):
    # UPDATE: 去重
    file2index = dict()
    ts_data_id, ts_data = [], []
    for i, path in enumerate(path_lst):
        for line in open(path, encoding='utf-8'):
            arr = line.strip().split('\t')
            id_ = arr[0]
            labels = [name2idx[name] for name in arr[3:]]
            file2index[id_] = labels
            # load ts data
            ts_data_id.append([config.train_dir[i], id_])
            ts_data_tmp = pd.read_csv(os.path.join(config.train_dir[i], id_), sep=' ').values
            ts_data.append(ts_data_tmp)
    ts_data_id, ts_data = np.array(ts_data_id), np.array(ts_data)
    _, unique_idx, inverse_idx = np.unique(ts_data, return_index=True, return_inverse=True, axis=0)
    unique_id = ts_data_id[unique_idx]
    duplicated = {i: [] for i in range(len(unique_idx))}
    for i in range(len(inverse_idx)):
        duplicated[inverse_idx[i]].append(tuple(ts_data_id[i]))
    duplicated = {k: v for k, v in duplicated.items() if len(v) > 1}

    # 去除掉标签不同的重复样本
    to_drop = set()
    for k, v in duplicated.items():
        drop = 0
        arry = file2index[v[0][1]]
        for i in range(1, len(v)):
            if file2index[v[i][1]] != arry:
                drop = 1
                break
        if drop:
            for x in v:
                to_drop.add(x[1])
    import pickle
    pickle.dump(duplicated, open('../user_data/duplicated_id.pkl', 'wb'))
    return {tuple(id_pair): file2index[id_pair[1]] for id_pair in unique_id if id_pair[1] not in to_drop}


def get_count(groups, dd=None):
    if dd is None:
        dd = torch.load(config.train_data)
    cnt = {name: 0 for name in dd['idx2name'].values()}
    for _, arrys in dd['file2idx'].items():
        for idx in arrys:
            cnt[dd['idx2name'][idx]] += 1
    res = {'multi':{k: [cnt[arry] for arry in v] for k, v in groups['multi'].items()},
            'solo': [cnt[arry] for arry in groups['solo']]}
    return res


def generate_targets(groups, file2idx, name2idx):
    res = {}
    for id_pair, arrys_index in file2idx.items():
        arrys_index = set(arrys_index)
        long_part = []
        for k in sorted(groups['multi'].keys()):
            for i, name in enumerate(groups['multi'][k]):
                if name2idx[name] in arrys_index:
                    tmp = i
                    break
            else:
                tmp = len(groups['multi'][k])
            long_part.append(tmp)
        float_part = [0.0]*len(groups['solo'])
        for i, name in enumerate(groups['solo']):
            if name2idx[name] in arrys_index:
                float_part[i] = 1.0
        res[id_pair[1]] = [long_part, float_part]
    return res


def get_train():
    name2idx = name2index()
    idx2name = {idx: name for name, idx in name2idx.items()}
    file2idx = file2label(config.train_label, name2idx)
    train, val = split_data(file2idx)
    dd = {'train': train, 'val': val, 'idx2name': idx2name, 'file2idx': file2idx}
    # count
    groups = utils.get_groups()
    dd['count'] = get_count(dd)
    # targets
    targets = generate_targets(groups, file2idx, name2idx)
    dd['targets'] = targets

    torch.save(dd, config.train_data)


def get_test():
    test_index = []
    for line in open(config.test_label, encoding='utf-8'):
        arr = line.strip().split('\t')
        id_ = arr[0]
        test_index.append((config.test_dir, id_))
    name2idx = name2index()
    idx2name = {idx: name for name, idx in name2idx.items()}
    dd = {'test': test_index, "idx2name": idx2name, 'file2idx': None, 'count': None, 'targets': None}
    torch.save(dd, config.test_data)


if __name__ == '__main__':
    # print('开始处理训练数据集...')
    # name2idx = name2index(config.arrythmia)
    # idx2name = {idx: name for name, idx in name2idx.items()}
    # get_train(name2idx, idx2name)
    # print('处理完成！')
    # print('开始处理测试数据集...')
    # get_test()
    # print('处理完成！')
    groups = utils.get_groups()
    dd = torch.load(config.train_data)
    file2idx = dd['file2idx']
    idx2name = dd['idx2name']
    name2idx = {v: k for k, v in idx2name.items()}
    targets = generate_targets(groups, file2idx, name2idx)
    pass




