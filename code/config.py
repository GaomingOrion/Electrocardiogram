# -*- coding: utf-8 -*-
import os

def get_groups(arrys):
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


class Config:
    # for data_process.py
    root = r'../data'
    train_dir = [os.path.join(root, 'train'), os.path.join(root, 'testA')]
    test_dir = os.path.join(root, 'testB_noDup_rename')
    train_label = [os.path.join(root, 'hf_round1_label.txt'), os.path.join(root, 'hefei_round1_ansA_20191008.txt')]
    test_label = os.path.join(root, 'hf_round1_subB_noDup_rename.txt')
    arrythmia = os.path.join(root, 'hf_round1_arrythmia.txt')
    train_data = '../user_data/train.pth'
    test_data = '../user_data/testB.pth'

    arrys = open(arrythmia, encoding='utf-8').read().rstrip().split()
    name2idx = {name: i for i, name in enumerate(arrys)}
    idx2name = {v: k for k, v in name2idx.items()}
    groups = get_groups(arrys)

    # for train
    #样本数
    num_samples = 32142
    #训练的模型名称
    model_name = 'resnet101'
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [20, 25, 50]
    #训练时的batch大小
    batch_size = 32
    #label的类别数
    num_classes = 63
    #最大训练多少个epoch
    max_epoch = 35
    #输入维度
    input_dim = 8
    #目标的采样长度
    target_point_num = 2048
    #保存模型的文件夹
    ckpt = '../user_data/ckpt'
    #保存提交文件的文件夹
    sub_dir = '../prediction_result'
    #初始的学习率
    lr = 1e-4
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10


config = Config()
