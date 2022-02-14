# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:57:15 2020

@author: Lim
"""
import os
import torch
from tqdm import tqdm
import numpy as np
from Loss import CtdetLoss
from torch.utils.data import DataLoader
from dataset import ctDataset
import random
from tensorboardX import SummaryWriter

from backbone.dlanet_dcn import DlaNet


def seed_torch(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # seed_torch()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_gpu = torch.cuda.is_available()

    write = SummaryWriter('./results/train/temp/')

    model = DlaNet(34)

    loss_weight = {'hm_weight': 1, 'ab_weight': 0.1, 'ang_weight': 0.1, 'reg_weight': 0.1,
                   'iou_weight': 15, 'mask_weight': 1}
    criterion = CtdetLoss(loss_weight)

    device = torch.device("cuda")
    print(device)

    if use_gpu:
        model.cuda()

    num_epochs = 150
    batch_size = 12

    optimizer = torch.optim.Adam(model.parameters(), lr=1.25e-4, eps=1e-8, amsgrad=True)

    train_dataset = ctDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    val_dataset = ctDataset(split='val')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    print('the dataset has %d images' % (len(train_dataset)))

    num_iter = 0

    best_test_loss = np.inf
    best_ap1 = 0
    best_ap2 = 0

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.

        pbar_dict = {'Loss': np.inf, 'Average_loss': np.inf,
                     'losses': [np.inf]}
        pbar = tqdm(total=len(train_dataset),
                    desc='[Epoch {:03d}]'.format(epoch + 1),
                    mininterval=0.1,
                    postfix=pbar_dict)
        # train
        for i, sample in enumerate(train_loader):
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)
            pred = model(sample['input'])

            loss, loss_show = criterion(pred, sample)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            write.add_scalar('Train Loss', total_loss / (i+1), global_step=epoch * len(train_loader) + i)
            write.add_scalar('hm Loss', loss_show[0], global_step=epoch * len(train_loader) + i)
            write.add_scalar('ab Loss', loss_show[1], global_step=epoch * len(train_loader) + i)
            write.add_scalar('ang Loss', loss_show[2], global_step=epoch * len(train_loader) + i)
            write.add_scalar('reg Loss', loss_show[3], global_step=epoch * len(train_loader) + i)
            write.add_scalar('iou Loss', loss_show[4], global_step=epoch * len(train_loader) + i)
            write.add_scalar('mask Loss', loss_show[5], global_step=epoch * len(train_loader) + i)
            write.add_image('train_mask', pred['mask'][0].sigmoid(), global_step=epoch * len(train_loader) + i)

            pbar_dict['Loss'] = loss.item()
            pbar_dict['Average_loss'] = total_loss / (i + 1)
            pbar_dict['losses'] = loss_show
            pbar.set_postfix(pbar_dict)
            pbar.update(batch_size)

        pbar.close()

        # validation
        validation_loss = 0.0
        model.eval()
        for i, sample in enumerate(val_loader):
            if use_gpu:
                for k in sample:
                    sample[k] = sample[k].to(device=device, non_blocking=True)
            pred = model(sample['input'])

            # all loss
            loss, _ = criterion(pred, sample)
            validation_loss += loss.item()
            write.add_image('val_mask', pred['mask'][0].sigmoid(), global_step=epoch + 1)

        validation_loss /= len(val_loader)

        write.add_scalar('val Loss', validation_loss, global_step=epoch + 1)


        # save model
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('* Best test loss %.5f' % best_test_loss)
            torch.save(model.state_dict(), './results/train/temp/best.pth')

        torch.save(model.state_dict(), './results/train/temp/epoch_{}.pth'.format(epoch+1))
        torch.save(model.state_dict(), './results/train/temp/last.pth')
