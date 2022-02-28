import argparse
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


def parse_opt():
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--data', type=str, default='data/GED', help='data folder path')
    parser.add_argument('--save_path', type=str, default='results/train', help='save path')
    parser.add_argument('--epoch', type=int, default=150, help='number of epoch')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU or not')

    # train parameters
    parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
    parser.add_argument('--val_batch_size', type=int, default=2, help='val batch size')
    parser.add_argument('--lr', type=float, default=1.25e-4, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon')
    parser.add_argument('--w_hm', type=float, default=1, help='weight of heatmap loss')
    parser.add_argument('--w_ab', type=float, default=0.1, help='weight of size(minor axis & major axis) loss')
    parser.add_argument('--w_ang', type=float, default=0.1, help='weight of angle loss')
    parser.add_argument('--w_reg', type=float, default=0.1, help='weight of offset loss')
    parser.add_argument('--w_iou', type=float, default=15, help='weight of Gaussian IoU loss')
    parser.add_argument('--w_mask', type=float, default=1, help='weight of mask loss')
    opt = parser.parse_args()
    return opt


def seed_torch(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(opt):
    save_path = opt.save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    write = SummaryWriter(save_path)
    # seed_torch()

    model = DlaNet(34)
    loss_weight = {'hm_weight': opt.w_hm, 'ab_weight': opt.w_ab, 'ang_weight': opt.w_ang, 'reg_weight': opt.w_reg,
                   'iou_weight': opt.w_iou, 'mask_weight': opt.w_mask}
    criterion = CtdetLoss(loss_weight)

    gpu_available = torch.cuda.is_available()
    if opt.use_gpu and gpu_available:
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
    print(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, eps=opt.eps, amsgrad=True)
    train_dataset = ctDataset(data_dir=opt.data, split='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_dataset = ctDataset(data_dir=opt.data, split='val')
    val_loader = DataLoader(val_dataset, batch_size=opt.val_batch_size, shuffle=False, num_workers=2)
    print('the train dataset has %d images' % (len(train_dataset)))
    print('the val dataset has %d images' % (len(val_dataset)))

    best_test_loss = np.inf
    for epoch in range(opt.epoch):
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

            write.add_scalar('Train Loss', total_loss / (i + 1), global_step=epoch * len(train_loader) + i)
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
            pbar.update(opt.batch_size)

        pbar.close()

        # validation
        validation_loss = 0.0
        model.eval()
        for i, sample in enumerate(val_loader):
            if opt.use_gpu:
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
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))

        torch.save(model.state_dict(), os.path.join(save_path, 'last.pth'))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    opt = parse_opt()
    train(opt)