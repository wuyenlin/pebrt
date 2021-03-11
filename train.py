#!/usr/bin/python3

from utils.options import args_parser
from common.petr_l import *
from common.dataloader import *
from common.loss import *


import sys
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time

def train(epoch, train_loader, val_loader, net, optimizer, scheduler):
    print("Training starts...")

    losses_3d_train = []
    losses_3d_valid = []

    # for ep in range(epoch):
    for ep in tqdm(range(epoch)):
        start_time = time()
        epoch_loss_3d_train = 0.0
        epoch_loss_3d_valid = 0.0
        N = 0
    # train
        for batch_idx, data in enumerate(train_loader,1):
            _, images, _, inputs_3d= data
            inputs_3d = inputs_3d.to(args.device)
            images = images.to(args.device)

            optimizer.zero_grad()

            predicted_3d_pos = net(images)

            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)
    # val
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader,1):
                _, images, _, inputs_3d= data
                inputs_3d = inputs_3d.to(args.device)
                images = images.to(args.device)

                optimizer.zero_grad()

                predicted_3d_pos = net(images)

                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0]*inputs_3d.shape[1]

            losses_3d_valid.append(epoch_loss_3d_valid / N)

        scheduler.step()
        elapsed = (time() - start_time)/60

        print('[%d] time %.2f 3d_train %f 3d_valid %f' % (
                ep + 1,
                elapsed,
                losses_3d_train[-1] * 1000,
                losses_3d_valid[-1]  *1000))
        if ep > 3:
            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig('loss_3d.png')

            plt.close('all')
    print('Finished Training.')
    return losses_3d_train , losses_3d_valid

if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PETR_L()
    net = net.to(args.device)

    model_params = 0
    for parameter in net.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    transforms = transforms.Compose([
        transforms.Resize([256,192]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Data("dataset/S1/Seq1/imageSequence/S1seq1.npz", transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=False, collate_fn=collate_fn)
    val_dataset = Data("dataset/S1/Seq1/imageSequence/S1seq1.npz", transforms, False)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=False, collate_fn=collate_fn)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_list, val_list = train(args.epoch, train_loader, val_loader, net, optimizer, scheduler)

    exp_name = "epoch_{}.bin".format(args.epoch)
    torch.save(net.state_dict(), exp_name)