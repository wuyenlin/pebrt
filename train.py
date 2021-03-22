#!/usr/bin/python3

from common.options import args_parser
from common.petr import *
from common.detr import *
from common.dataloader import *
from common.loss import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time


def train(epoch, train_loader, val_loader, model, optimizer, scheduler):
    print("Training starts...")

    losses_3d_train = []
    losses_3d_valid = []

    for ep in tqdm(range(epoch)):
        start_time = time()
        epoch_loss_3d_train = 0.0
        N = 0
        model.train()
    # train
        for data in train_loader:
            _, images, _, inputs_3d= data
            inputs_3d = inputs_3d.to(args.device)
            images = images.to(args.device)

            optimizer.zero_grad()

            predicted_3d_pos = model(images)

            loss_3d_pos = anth_mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)
    # val
        with torch.no_grad():
            model.load_state_dict(model.state_dict())
            model.eval()
            epoch_loss_3d_valid = 0.0
            N = 0

            for data in val_loader:
                _, images, _, inputs_3d= data
                inputs_3d = inputs_3d.to(args.device)
                images = images.to(args.device)

                optimizer.zero_grad()

                predicted_3d_pos = model(images)

                loss_3d_pos = anth_mpjpe(predicted_3d_pos, inputs_3d)
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
        if args.export_training_curves and ep > 3:
            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig('./checkpoint/loss_3d.png')

            plt.close('all')

        if (ep)%10 == 0 and ep != 0:
            exp_name = "./checkpoint/epoch_{}.bin".format(ep)
            torch.save(model.state_dict(), exp_name)
            print("Saved param")

    print('Finished Training.')
    return losses_3d_train , losses_3d_valid

if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PETR(lift=args.lift)
    model = model.to(args.device)
    if args.chkpt is not None:
        model.load_state_dict(torch.load(args.chkpt))
        print("INFO: Checkpoint loaded from {}".format(args.chkpt))

    if args.lift:
        print("INFO: Model loaded. Using Lifting model.")
        backbone_params = 0
        if args.freeze:
            print("INFO: Freezing HRNet")
            for param in model.backbone.parameters():
                param.requires_grad = False
                backbone_params += param.numel()
    else:
        print("INFO: Model loaded. Using End-to-end model.")

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    if args.lift and args.freeze:
        model_params -= backbone_params
    
    print("INFO: Trainable parameter count:", model_params, " (%.2f M)" %(model_params/1000000))

    transforms = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    train_dataset = Data("dataset/S1/Seq1/imageSequence/S1seq1.npz", transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=16, drop_last=True, collate_fn=collate_fn)
    val_dataset = Data("dataset/S1/Seq1/imageSequence/S1seq1.npz", transforms, False)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=16, drop_last=True, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    print("INFO: Using optimizer {}".format(optimizer))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_list, val_list = train(args.epoch, train_loader, val_loader, model, optimizer, scheduler)