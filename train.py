import yaml
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
from pathlib import Path
import torchmetrics

from dataset.CricketFrames import FramesLoader
from dataset.utils import batching_frame_seq
from metrics.classsification_metrics import metrics_init,metrics_op,metrics_compute,print_metrics,tensorboard_metric_plot
from models.CRNN_Model import Resnt18Rnn
from torch.utils.tensorboard import SummaryWriter

def train_epoch(model,train_loader,optimizer,criterion):
    model.train()
    losses=[]
    # train_acc=torchmetrics.Accuracy()
    metrices = metrics_init()
    for i, data in enumerate(train_loader):
        inp, gt_cls,_ = data
        inp,gt_cls = inp.cuda(),gt_cls.cuda()
        out = model(inp)
        loss = criterion(out,gt_cls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        metrics_op(metrices,out.cpu(),gt_cls.cpu())
        # train_acc(out.cpu(),gt_cls.cpu())    
    avg_losses = sum(losses)/len(losses)
    total_train_res = metrics_compute(metrices)
    return avg_losses,total_train_res
def val_epoch(model,val_loader,criterion):
    model.eval()
    losses=[]
    # val_acc=torchmetrics.Accuracy()
    metrices = metrics_init()
    for i, data in enumerate(val_loader):
        inp, gt_cls,_ = data
        inp,gt_cls = inp.cuda(),gt_cls.cuda()
        with torch.no_grad():
            out = model(inp)
            loss = criterion(out.cpu(),gt_cls.cpu())
        losses.append(loss.item())
        metrics_op(metrices,out.cpu(),gt_cls.cpu())
        # val_acc(out.cpu(),gt_cls.cpu())
    avg_losses = sum(losses)/len(losses)
    total_val_res = metrics_compute(metrices)
    # val_acc.reset()
    return avg_losses,total_val_res

def train(cfg):
    dataset = FramesLoader(cfg['dataset']['data_dir'])
    train_size = int(0.7 * (round(len(dataset))))  # split in 0.7,0.3 for train and test.
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,collate_fn=batching_frame_seq)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = Resnt18Rnn(num_classes=cfg['model']['num_classes'], pretrained=True, rnn_hidden_size=cfg['model']['cnnrnn']['rnn_hidden_size'], rnn_num_layers=cfg['model']['cnnrnn']['rnn_num_layers'])
    model = model.cuda()
    # critertion = nn.BCEWithLogitsLoss(reduction='none')  # loss function
    critertion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])  # Optimizer
    Path("./results/tensorboard/train").mkdir(parents=True, exist_ok=True)
    Path("./results/tensorboard/val").mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter("./results/tensorboard/train")
    val_writer = SummaryWriter("./results/tensorboard/val")
    for epoch in range(cfg['training']['epoch']):
        train_losses,train_res = train_epoch(model,train_loader,optimizer,critertion)
        tensorboard_metric_plot(train_writer,epoch,train_res,train_losses)
        val_losses,val_res = val_epoch(model,val_loader,critertion)
        tensorboard_metric_plot(val_writer,epoch,val_res,val_losses)
        print(f'training losses={train_losses}, validation losses = {val_losses}')
        # print(f'training acc={training_acc}, validation acc = {val_acc}')
if __name__ == '__main__':
    with open('./configs/cfg.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    train(cfg)
