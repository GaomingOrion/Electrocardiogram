#!/usr/bin/env python
# -*- coding: utf-8 -*-
from config import config
from data_process import get_train, get_test
from model_run import train, test

if __name__ == '__main__':
    # 数据处理
    print('开始处理训练数据集...')
    get_train()
    print('处理完成！')
    print('开始处理测试数据集...')
    get_test()
    print('处理完成！')

    # 训练
    print('开始训练，总epochs: %i'% config.max_epoch)
    train('train')
    print('训练完成')

    # 测试
    print('开始测试...')
    test()
    print('测试完成，已经生成提交结果')


