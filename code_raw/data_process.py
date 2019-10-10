# -*- coding: utf-8 -*-
import os, torch
import numpy as np
import pandas as pd
from config import config

# seed
np.random.seed(41)


def name2index(path):
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2idx = {name: i for i, name in enumerate(list_name)}
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


def file2index(path_lst, name2idx):
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


def get_train(name2idx, idx2name):
    file2idx = file2index(config.train_label, name2idx)
    print(len(file2idx))
    train, val = split_data(file2idx)
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx, 'wc': None}
    dd['wc'] = get_proption(dd, True)
    torch.save(dd, config.train_data)


def get_test():
    test_index = []
    for line in open(config.test_label, encoding='utf-8'):
        arr = line.strip().split('\t')
        id_ = arr[0]
        test_index.append((config.test_dir, id_))
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    dd = {'test': test_index, "idx2name": idx2name, 'file2idx': None, 'wc': None}
    torch.save(dd, config.test_data)


def get_proption(dd=None, return_count=False):
    if dd is None:
        dd = torch.load(config.train_data)
    n = len(dd['train']) + len(dd['val'])
    cnt = [0]*55
    for id_ in dd['train']:
        for idx in dd['file2idx'][id_]:
            cnt[idx] += 1
    for id_ in dd['val']:
        for idx in dd['file2idx'][id_]:
            cnt[idx] += 1
    if return_count:
        return cnt
    prop = np.array(cnt) / n
    np.save('../user_data/proption_big.npy', prop)
    return prop

def get_groups():
    arrys = open(config.arrythmia, encoding='utf-8').read().rstrip().split()
    groups = [['窦性心动过缓', '窦性心动过速', '窦性心律'],
             ['正常ECG', '临界ECG', '异常ECG'],
             ['左心室肥大', '左心室高电压', 'QRS低电压'],
             ['快心室率', '慢心室率'],
             ['T波改变', 'ST段改变', 'ST-T改变'],
             ['非特异性T波异常', '非特异性ST段异常', '非特异性ST段与T波异常'],
             ['电轴左偏', '电轴右偏'],
             ['右束支传导阻滞', '完全性右束支传导阻滞', '不完全性右束支传导阻滞'],
             ['左束支传导阻滞', '完全性左束支传导阻滞', '左前分支传导阻滞']]
    groups_set = set()
    for group in groups:
        for x in group:
            groups_set.add(x)
    left = [x for x in arrys if x not in groups_set and x != '双分支传导阻滞']
    return groups, left

if __name__ == '__main__':
    print('开始处理训练数据集...')
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    get_train(name2idx, idx2name)
    print('处理完成！')
    print('开始处理测试数据集...')
    get_test()
    print('处理完成！')


