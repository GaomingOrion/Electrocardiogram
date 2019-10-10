# -*- coding: utf-8 -*-
'''
@time: 2019/7/23 19:42

@ author: javis
'''
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
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


def prob2pred(prob):
    return np.where(prob > 0.5, 1, 0)
    # proption = np.load('./data/proption.npy')
    # n = len(prob)
    # count = [int(n * proption[i]) for i in range(55)]
    # print(count)
    # preds = np.zeros([n, 55])
    # for i in range(55):
    #     tmp = np.argsort(prob[:, i])[(n-count[i]):]
    #     preds[tmp, i] = 1
    # return preds


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    ys, probs = [], []
    loss_lst = []
    it_count = 0
    for inputs, extra_info, target in train_dataloader:
        it_count += 1
        inputs = inputs.to(device)
        extra_info = extra_info.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs, extra_info)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        ys.append(target.data.cpu().numpy())
        loss_lst.append(loss.item())
        prob = torch.sigmoid(output).detach().cpu().numpy()
        probs.append(prob)
        if it_count != 0 and it_count % show_interval == 0:
            print("step-%d,loss:%.3e" % (it_count, loss.item()))
    ys = np.concatenate(ys)
    probs = np.concatenate(probs)
    preds = prob2pred(probs)
    recall = recall_score(ys, preds, average='micro')
    precision = precision_score(ys, preds, average='micro')
    f1 = 2 * recall * precision / (recall + precision)
    train_loss = sum(loss_lst) / len(loss_lst)
    return train_loss, precision, recall, f1


def val_epoch(model, criterion, val_dataloader, threshold=0.5, simple_mode=True):
    model.eval()
    ys, probs = [], []
    loss_lst = []
    with torch.no_grad():
        for inputs, extra_info, target in val_dataloader:
            inputs = inputs.to(device)
            extra_info = extra_info.to(device)
            ys.append(target.data.numpy())
            target = target.to(device)
            output = model(inputs, extra_info)
            loss = criterion(output, target)
            loss_lst.append(loss.item())
            prob = torch.sigmoid(output).detach().cpu().numpy()
            probs.append(prob)
    ys = np.concatenate(ys)
    probs = np.concatenate(probs)
    preds = prob2pred(probs)
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
    model = getattr(resnet, config.model_name)(input_dim=config.input_dim)
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
    weights = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(weights)
    # criterion = torch.nn.BCEWithLogitsLoss()
    # 模型保存文件夹
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if resume:
        if os.path.exists(ckpt):  # 这里是存放权重的目录
            model_save_dir = ckpt
            current_w = torch.load(os.path.join(ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            # 如果中断点恰好为转换stage的点
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    if not os.path.exists(config.ckpt):
        os.mkdir(config.ckpt)
    os.mkdir(model_save_dir)
    # =========>开始训练<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_p, train_r, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=50)
        val_loss, val_p, val_r, val_f1 = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d time:%s' % (epoch, stage, utils.print_time_cost(since)))
        print('train_loss:%.3e train_precision:%.4f train_recall:%.4f train_f1:%.4f' % (train_loss, train_p, train_r, train_f1))
        print('val_loss:%.3e val_precision:%.4f val_recall:%.4f val_f1:%.4f \n' % (val_loss, val_p, val_r, val_f1))
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
    threshold = 0.5
    model = getattr(resnet, config.model_name)(input_dim=config.input_dim)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, mode=mode)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    val_loss, val_p, val_r, val_f1, pr_df = val_epoch(model, criterion, val_dataloader, threshold, False)
    print('threshold %.2f val_loss:%0.3e val_precision:%.4f val_recall:%.4f val_f1:%.4f\n'
          % (threshold, val_loss, val_p, val_r, val_f1,))
    pr_df['arry'] = pr_df['arry'].map(val_dataset.idx2name)
    pr_df.to_csv('../user_data/%s_f1.csv' % mode, encoding='gbk')
    print(pr_df)


def test():
    test_dataset = ECGDataset(data_path=config.test_data, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=6)
    # model
    model = getattr(resnet, config.model_name)(input_dim=config.input_dim)
    model = model.to(device)
    ckpt_dir = os.path.join(config.ckpt, os.listdir(config.ckpt)[0])
    ckpts = [os.path.join(ckpt_dir, 'e%i'%i) for i in range(25, 30)]

    # make prediction
    preds_ckpts = []
    for ckpt in ckpts:
        model.load_state_dict(torch.load(ckpt)['state_dict'])
        model.eval()
        probs = []
        with torch.no_grad():
            for inputs, extra_info in test_dataloader:
                inputs = inputs.to(device)
                extra_info = extra_info.to(device)
                output = model(inputs, extra_info)
                prob = torch.sigmoid(output).detach().cpu().numpy()
                probs.append(prob)
        probs = np.concatenate(probs)
        preds = prob2pred(probs)
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
    train('train')