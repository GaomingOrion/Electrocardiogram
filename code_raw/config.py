# -*- coding: utf-8 -*-
import os


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

    # for train
    #训练的模型名称
    model_name = 'resnet34'
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [20, 25, 50]
    #训练时的batch大小
    batch_size = 64
    #label的类别数
    num_classes = 55
    #最大训练多少个epoch
    max_epoch = 29
    #输入维度
    input_dim = 10
    #目标的采样长度
    target_point_num = 2048
    #保存模型的文件夹
    ckpt = '../user_data/ckpt'
    #保存提交文件的文件夹
    sub_dir = '../prediction_result'
    #初始的学习率
    lr = 1e-3
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10


config = Config()
