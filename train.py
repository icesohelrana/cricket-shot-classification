import yaml
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn

from dataset.CricketFrames import FramesLoader
from models.CRNN_Model import Resnt18Rnn
from torch.utils.tensorboard import SummaryWriter

def train_epoch(model,train_loader,optimizer,criterion):
    model.train()
    losses=[]
    for i, data in enumerate(train_loader):
        inp, gt_cls,quality = data
        out = model(inp)
        loss = criterion(out,gt_cls)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    avg_losses = sum(losses)/len(losses)
    return avg_losses

def train(cfg):
    dataset = FramesLoader(cfg['dataset']['data_dir'],cfg['dataset']['ann_dir'])
    train_size = int(0.7 * (round(len(dataset))))  # split in 0.7,0.3 for train and test.
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = Resnt18Rnn(num_classes=cfg['model']['num_classes'], pretrained=True, rnn_hidden_size=cfg['model']['cnnrnn']['rnn_hidden_size'], rnn_num_layers=cfg['model']['cnnrnn']['rnn_num_layers'])
    # critertion = nn.BCEWithLogitsLoss(reduction='none')  # loss function
    critertion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])  # Optimizer
    for epoch in range(cfg['training']['epoch']):
        losses = train_epoch(model,train_loader,optimizer,critertion)
        print(losses)
if __name__ == '__main__':
    with open('./configs/cfg.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    train(cfg)
