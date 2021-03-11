#!/usr/bin/python3

from utils.options import args_parser
from common.hrnet import *
from common.dataloader import *
from common.loss import *


import sys
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time

def train(epoch, train_loader, val_loader, net, optimizer, scheduler):
    print("Training starts...")

    losses_2d_train = []
    losses_2d_valid = []

    # for ep in range(epoch):
    for ep in tqdm(range(epoch)):
        start_time = time()
        epoch_loss_2d_train = 0.0

        N = 0
    # train
        for batch_idx, data in enumerate(train_loader,1):
            _, images, inputs_2d, _ = data
            inputs_2d = inputs_2d.to(args.device)
            images = images.to(args.device)

            optimizer.zero_grad()

            predicted_2d_pos = net(images)

            loss_2d_pos = mpjpe(predicted_2d_pos, inputs_2d)
            epoch_loss_2d_train += inputs_2d.shape[0]*inputs_2d.shape[1] * loss_2d_pos.item()
            N += inputs_2d.shape[0]*inputs_2d.shape[1]

            loss_total = Variable(loss_2d_pos, requires_grad=True)
            loss_total.backward()

            optimizer.step()

        losses_2d_train.append(epoch_loss_2d_train / N)
    # val
        with torch.no_grad():
            epoch_loss_2d_valid = 0.0
            N = 0
            for batch_idx, data in enumerate(val_loader,1):
                _, images, inputs_2d, _ = data
                inputs_2d = inputs_2d.to(args.device)
                images = images.to(args.device)

                optimizer.zero_grad()

                predicted_2d_pos = net(images)

                loss_2d_pos = mpjpe(predicted_2d_pos, inputs_2d)
                epoch_loss_2d_valid += inputs_2d.shape[0]*inputs_2d.shape[1] * loss_2d_pos.item()
                N += inputs_2d.shape[0]*inputs_2d.shape[1]

            losses_2d_valid.append(epoch_loss_2d_valid / N)

        scheduler.step()
        elapsed = (time() - start_time)/60

        print('[%d] time %.2f 2d_train %f 2d_valid %f' % (
                ep + 1,
                elapsed,
                losses_2d_train[-1] * 1000,
                losses_2d_valid[-1]  *1000))
        if ep > 3:
            plt.figure()
            epoch_x = np.arange(3, len(losses_2d_train)) + 1
            plt.plot(epoch_x, losses_2d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_2d_valid[3:], color='C1')
            plt.legend(['2d train', '2d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig('loss_2d.png')

            plt.close('all')
    print('Finished Training.')
    return losses_2d_train , losses_2d_valid

if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = HRNet(32, 17, 0.1)
    net = net.to(args.device)
    net.load_state_dict(torch.load('./weights/pose_hrnet_w32_256x192.pth'))
    print("Pretrained weights loaded!")

    model_params = 0
    for parameter in net.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    train_dataset = Data("dataset/S1/Seq1/imageSequence/full_S1seq1.npz", transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=16, drop_last=False, collate_fn=collate_fn)
    val_dataset = Data("dataset/S1/Seq1/imageSequence/full_S1seq1.npz", transforms, False)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=16, drop_last=False, collate_fn=collate_fn)

    # optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_list, val_list = train(args.epoch, train_loader, val_loader, net, optimizer, scheduler)

    exp_name = "finetune_epoch_{}.bin".format(args.epoch)
    torch.save(net.state_dict(), exp_name)