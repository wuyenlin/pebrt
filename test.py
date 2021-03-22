#!/usr/bin/python3

from common.options import args_parser
from common.petr import *
from common.dataloader import *
from common.loss import *

import matplotlib.pyplot as plt
from numpy import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from time import time


def evaluate(test_loader, model):
    print("Testing starts...")
    epoch_loss_3d_pos = 0.0
    epoch_loss_3d_pos_procrustes = 0.0
    epoch_loss_3d_pos_scale = 0.0

    with torch.no_grad():
        model.eval()
        N = 0
        for data in test_loader:
            _, images, _, inputs_3d= data
            inputs_3d = inputs_3d.to(args.device)
            images = images.to(args.device)

            predicted_3d_pos = model(images)
            
            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000

    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('----------')
    
    return e1, e2, e3



if __name__=="__main__":
    args = args_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PETR(lift=args.lift)
    model = model.to(args.device)
    if args.chkpt is not None:
        model.load_state_dict(torch.load(args.chkpt))
        print("INFO: Checkpoint loaded from {}".format(args.chkpt))

    transforms = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    test_dataset = Data("dataset/S1/Seq1/imageSequence/S1seq1.npz", transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=8, collate_fn=collate_fn)
    e1, e2, e3 = evaluate(test_loader, model)