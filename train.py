#!/usr/bin/python3

from utils.options import args_parser
from utils.dataloader import *
from utils.model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime

def train(epoch, train_loader, val_loader, net, criterion, optimizer, scheduler):
    print("Training starts...")

    train_loss_list = []
    val_loss_list = []

    training_loss = 0.0
    validation_loss = 0.0

    for ep in range(epoch):
    # train
        for batch_idx, data in enumerate(train_loader,1):
            _, images, vecs = data
            vecs = [eval(item) for item in vecs]
            vecs = torch.Tensor(vecs)
            vecs = vecs.to(args.device)
            images = images.to(args.device)

            outputs = net(images)
            loss = criterion(outputs, vecs)
            loss = torch.sqrt(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            num = 15000//args.bs + 1
            if batch_idx%num == 0:
                training_loss /= num
                train_loss_list.append(training_loss)
                f_train.write(str(training_loss) + "\n")
                print("Train\tEpoch {}.\tBatch {}.\tLoss = {:.3f}.".format(ep+1, batch_idx, training_loss))
                training_loss = 0.0
    # val
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader,1):
                _, images, vecs = data
                vecs = [eval(item) for item in vecs]
                vecs = torch.Tensor(vecs)
                vecs = vecs.to(args.device)
                images = images.to(args.device)

                outputs = net(images)
                
                loss = criterion(vecs, outputs)
                # loss = torch.sqrt(loss)

                optimizer.zero_grad()
                
                validation_loss += loss.item()
                num = 5000//args.bs + 1
                if batch_idx%num == 0:
                    validation_loss /= num
                    val_loss_list.append(validation_loss)
                    f_val.write(str(validation_loss) + "\n")
                    print("Val\tEpoch {}.\tBatch {}.\tLoss = {:.3f}.\n".format(ep+1, batch_idx, validation_loss))
                    validation_loss = 0.0
        scheduler.step()
    print('Finished Training.')
    return train_loss_list, val_loss_list

if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.pretrained:
        for param in net.model.parameters():
            param.requires_grad = args.req_grad
    net = net.to(args.device)

    train_dataset = selfData("dataset/train_15000.txt", transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=16, drop_last=False, collate_fn=collate_fn)
    val_dataset = selfData("dataset/val_5000.txt", transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=16, drop_last=False, collate_fn=collate_fn)

    optimizer = optim.SGD(net.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss(reduction="mean")

    train_list, val_list = train(args.epoch, train_loader, val_loader, net, criterion, optimizer, scheduler)

    exp_name = "log/{}/resnet_e50_bs{}_lr{}".format(args.sess, args.bs, str(args.lr).replace(".",""))
    PATH = exp_name + ".pth"
    torch.save(net.state_dict(), PATH)