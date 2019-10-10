# -*- coding: utf-8 -*-
'''
@time: 2019/7/23 19:42

@ author: javis
'''
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
import os, time
import utils
from dataset import ECGDataset
import resnet
from config import config

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)


def output2pred(outputs):
    groups = config.groups
    name2idx = config.name2idx
    n = len(outputs)
    keys = list(sorted(groups['multi'].keys()))
    output_index = [0]
    for k in keys:
        output_index.append(output_index[-1] + len(groups['multi'][k]) + 1)
    preds = np.zeros([n, 55])
    for i in range(n):
        for j, k in enumerate(keys):
            max_index = np.argmax(outputs[i, output_index[j]:output_index[j+1]])
            if max_index != len(groups['multi'][k]):
                preds[i, name2idx[groups['multi'][k][max_index]]] = 1
    for j, k in enumerate(groups['solo']):
        preds[:, name2idx[k]] = np.where(outputs[:, output_index[-1]+j] > 0, 1, 0)
    preds[:, 50] = np.where(preds[:, 19] + preds[:, 25] == 2, 1, 0)
    return preds


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    ys, outputs = [], []
    loss_lst = []
    it_count = 0
    for inputs, extra_info, target_long, target_float, target in train_dataloader:
        it_count += 1
        inputs = inputs.to(device)
        extra_info = extra_info.to(device)
        target_long = target_long.to(device)
        target_float = target_float.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs, extra_info)
        loss = criterion(output, target_long, target_float)
        loss.backward()
        optimizer.step()
        ys.append(target.data.cpu().numpy())
        loss_lst.append(loss.item())
        outputs.append(output.detach().cpu().numpy())
        if it_count != 0 and it_count % show_interval == 0:
            print("step-%d,loss:%.3e" % (it_count, loss.item()))
    ys = np.concatenate(ys)
    outputs = np.concatenate(outputs)
    preds = output2pred(outputs)
    recall = recall_score(ys, preds, average='micro')
    precision = precision_score(ys, preds, average='micro')
    f1 = 2 * recall * precision / (recall + precision)
    train_loss = sum(loss_lst) / len(loss_lst)
    return train_loss, precision, recall, f1


def val_epoch(model, criterion, val_dataloader, simple_mode=True):
    model.eval()
    ys, outputs = [], []
    loss_lst = []
    with torch.no_grad():
        for inputs, extra_info, target_long, target_float, target in val_dataloader:
            inputs = inputs.to(device)
            extra_info = extra_info.to(device)
            target_long = target_long.to(device)
            target_float = target_float.to(device)
            ys.append(target.data.numpy())
            output = model(inputs, extra_info)
            loss = criterion(output, target_long, target_float)
            loss_lst.append(loss.item())
            outputs.append(output.detach().cpu().numpy())
    ys = np.concatenate(ys)
    outputs = np.concatenate(outputs)
    preds = output2pred(outputs)
    recall = recall_score(ys, preds, average='micro')
    precision = precision_score(ys, preds, average='micro')
    f1 = 2*recall*precision/(recall + precision)
    val_loss = sum(loss_lst) / len(loss_lst)
    if simple_mode:
        return val_loss, precision, recall, f1
    else:
        df = []
        for i in range(55):
            true = np.sum(ys[:, i])
            pred_true = np.sum(preds[:, i])
            all_true = np.sum((ys[:, i] == 1) & (preds[:, i] == 1))
            recall_one = recall_score(ys[:, i], preds[:, i])
            precision_one = precision_score(ys[:, i], preds[:, i])
            f1_one = 2 * recall_one * precision_one / (recall_one + precision_one + 1e-8)
            df.append([i, true, pred_true, all_true, recall_one, precision_one, f1_one])
        df = pd.DataFrame(df, columns=['arry', 'true', 'pred_true', 'all_true', 'recall', 'precision', 'f1'])
        return val_loss, precision, recall, f1, df


def train(mode='train', ckpt=None, resume=False):
    # model
    model = getattr(resnet, config.model_name)(num_classes=config.num_classes, input_dim=config.input_dim)
    if ckpt is not None and not resume:
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    train_dataset = ECGDataset(data_path=config.train_data, mode=mode)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_dataset = ECGDataset(data_path=config.train_data, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    groups = utils.get_groups()
    count = train_dataset.count
    criterion = utils.WeightedMultilabel(groups, count, device)
    # 模型保存文件夹
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if resume:
        if os.path.exists(ckpt):  # 这里是存放权重的目录
            current_w = torch.load(os.path.join(ckpt))
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    val_loss, val_p, val_r, val_f1 = val_epoch(model, criterion, val_dataloader)
    print('start training')
    print('val_loss:%.3e val_precision:%.4f val_recall:%.4f val_f1:%.4f \n' % (val_loss, val_p, val_r, val_f1))
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_p, train_r, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=50)
        val_loss, val_p, val_r, val_f1 = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d time:%s' % (epoch, stage, utils.print_time_cost(since)))
        print('train_loss:%.3e train_precision:%.4f train_recall:%.4f train_f1:%.4f' % (train_loss, train_p, train_r, train_f1))
        print('val_loss:%.3e val_precision:%.4f val_recall:%.4f val_f1:%.4f \n' % (val_loss, val_p, val_r, val_f1))
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}
        torch.save(state, os.path.join(model_save_dir, 'e%i' % (epoch)))
        best_f1 = max(best_f1, val_f1)

        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)


def val(mode, ckpt):
    model = getattr(resnet, config.model_name)(num_classes=config.num_classes, input_dim=config.input_dim)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    val_dataset = ECGDataset(data_path=config.train_data, mode=mode)
    groups = utils.get_groups()
    count = val_dataset.count
    criterion = utils.WeightedMultilabel(groups, count, device)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    val_loss, val_p, val_r, val_f1, pr_df = val_epoch(model, criterion, val_dataloader, False)
    print('val_loss:%0.3e val_precision:%.4f val_recall:%.4f val_f1:%.4f\n'
          % (val_loss, val_p, val_r, val_f1,))
    pr_df['arry'] = pr_df['arry'].map(val_dataset.idx2name)
    pr_df.to_csv('../user_data/%s_f1.csv' % mode, encoding='gbk')
    print(pr_df)


def test():
    test_dataset = ECGDataset(data_path=config.test_data, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=6)
    # model
    model = getattr(resnet, config.model_name)(num_classes=config.num_classes, input_dim=config.input_dim)
    model = model.to(device)
    ckpt_dir = os.path.join(config.ckpt, os.listdir(config.ckpt)[0])
    # ckpts = [os.path.join(ckpt_dir, 'e%i'%i) for i in range(25, 30)]
    ckpts = [os.path.join('../user_data/ckpt/resnet34_201910101016', 'e%i'%i) for i in range(31, 35)]

    # make prediction
    preds_ckpts = []
    for ckpt in ckpts:
        model.load_state_dict(torch.load(ckpt)['state_dict'])
        model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, extra_info in test_dataloader:
                inputs = inputs.to(device)
                extra_info = extra_info.to(device)
                output = model(inputs, extra_info)
                outputs.append(output.detach().cpu().numpy())
        outputs = np.concatenate(outputs)
        preds = output2pred(outputs)
        preds_ckpts.append(preds)
    preds = np.median(np.array(preds_ckpts), axis=0)
    preds_dict = {id_[1]: preds[i] for i, id_ in enumerate(test_dataset.data)}

    # make submission
    sub_file = os.path.join(config.sub_dir, 'result.txt')
    fout = open(sub_file, 'w', encoding='utf-8')
    for line in open(config.test_label, encoding='utf-8'):
        fout.write(line.strip('\n'))
        id_ = line.split('\t')[0]
        pred = preds_dict[id_]
        ixs = [i for i, out in enumerate(pred) if out == 1]
        for i in ixs:
            fout.write("\t" + test_dataset.idx2name[i])
        fout.write('\n')
    fout.close()


if __name__ == '__main__':
    # train('train')
    # train('all', '../user_data/ckpt/resnet34_201910100416/e28', True)
    # val('val', '../user_data/ckpt/resnet34_201910100416/e28')
    # test()
    train('train')
    pass