#!/usr/bin/python3

from common.options import args_parser
from common.petr import *
from common.peltra import *
from common.dataloader import *
from common.loss import *

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


def evaluate(test_loader, model, device):
    print("Testing starts...")
    epoch_loss_3d_pos = 0.0

    with torch.no_grad():
        model.eval()
        N = 0
        for data in test_loader:
            _, images, inputs_2d, vec_3d = data
            inputs_2d = inputs_2d.to(device)
            vec_3d = vec_3d.to(device)
            images = images.to(device)

            predicted_3d_pos = model(inputs_2d)

            error = maev(predicted_3d_pos, vec_3d)
            epoch_loss_3d_pos += vec_3d.shape[0]*vec_3d.shape[1] * error.item()
            
    e1 = (epoch_loss_3d_pos / N)

    print('----------')
    print('Protocol #1 Error (MAEV):', e1)
    print('Protocol #2 Error (L2 Norm):', e2)
    print('Protocol #3 Error (Euler angles):', e3)
    print('----------')
    
    return e1


def main(args):

    device = torch.device(args.device)
    model = PELTRA(device)
    model = model.to(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    print("INFO: Model loaded on {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    
    print("INFO: Trainable parameter count:", model_params, " (%.2f M)" %(model_params/1000000))

    print("INFO: Evaluation Mode")
    test_dataset = Data(args.dataset, transforms, False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    e1, e2, ev = evaluate(test_loader, model, device)
    return e1, e2, ev

    # param_dicts = [
    #     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]

    # optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    # print("INFO: Using optimizer {}".format(optimizer))

    # train_list, val_list = train(args.start_epoch, args.epoch, 
    #                             train_loader, val_loader, model, device,
    #                             optimizer, lr_scheduler)


if __name__ == "__main__":
    args = args_parser()
    main(args)
