#!/usr/bin/python3

from common.peltra import *
from common.dataloader import *
from common.loss import *
from common.human import *

import argparse
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
import random


parser = argparse.ArgumentParser("Set PEBRT parameters", add_help=False)

# Hyperparameters
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--bs", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-04)
parser.add_argument("--lr_backbone", type=float, default=0)
parser.add_argument("--weight_decay", type=float, default=1e-05)
parser.add_argument("--lr_drop", default=10, type=int)

# Transformer (layers of enc and dec, dropout rate, num_heads, dim_feedforward)
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate applied in transformer")

# dataset
parser.add_argument("--num_workers", default=1, type=int)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--export_training_curves", action="store_true", help="Save train/val curves in .png file")
# parser.add_argument("--dataset", type=str, default="./dataset/S1/Seq1/imageSequence/S1.npz")
parser.add_argument("--dataset", type=str, default="./h36m/data_h36m_frame_all.npz")
parser.add_argument("--device", default="cuda", help="device used")
parser.add_argument("--resume", type=str, default=None, help="Loading model checkpoint")
parser.add_argument("--distributed", action="store_true")

# SLI
parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
parser.add_argument("--random_seed", type=int, help="Random seed.", default=0)

args = parser.parse_args()


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


def train(start_epoch, epoch, train_loader, val_loader, 
            model, device, optimizer, lr_scheduler, local_rank):
    print("Training starts...")

    losses_3d_train = []
    losses_3d_valid = []

    for ep in range(start_epoch, epoch):
        start_time = time()
        epoch_loss_3d_train = 0.0
        N = 0

        if ep%5 == 0 and ep != 0:
            if local_rank == 0:
                exp_name = "./peltra/epoch_{}.bin".format(ep)
                torch.save({
                    "epoch": ep,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model": model.state_dict(),
                    "args": args,
                }, exp_name)
                print("Parameters saved to ", exp_name)

        model.train()
    # train
        for data in train_loader:
            _, image, inputs_2d, vec_3d = data
            inputs_2d = inputs_2d.to(device)
            vec_3d = vec_3d.to(device)

            optimizer.zero_grad()

            predicted_3d_pos, w_kc = model(inputs_2d)

            loss_3d_pos = maev(predicted_3d_pos, vec_3d, w_kc)
            epoch_loss_3d_train += vec_3d.shape[0] * loss_3d_pos.item()
            N += vec_3d.shape[0]

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
                _, image, inputs_2d, vec_3d = data
                inputs_2d = inputs_2d.to(device)
                vec_3d = vec_3d.to(device)

                predicted_3d_pos, w_kc = model(inputs_2d)

                loss_3d_pos = maev(predicted_3d_pos, vec_3d, w_kc)
                epoch_loss_3d_valid += vec_3d.shape[0] * loss_3d_pos.item()
                N += vec_3d.shape[0]

            losses_3d_valid.append(epoch_loss_3d_valid / N)

        lr_scheduler.step()
        elapsed = (time() - start_time)/60

        if local_rank == 0:
            print("[%d] time %.2f 3d_train %f 3d_valid %f" % (
                    ep + 1,
                    elapsed,
                    losses_3d_train[-1] * 1000,
                    losses_3d_valid[-1] * 1000))

        if args.export_training_curves and ep > 3:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], "--", color="C0")
            plt.plot(epoch_x, losses_3d_valid[3:], color="C1")
            plt.legend(["3d train", "3d valid (eval)"])
            plt.ylabel("MPJPE (m)")
            plt.xlabel("Epoch")
            plt.xlim((3, epoch))
            plt.savefig("./peltra/loss_3d.png")

            plt.close("all")

    print("Finished Training.")
    return losses_3d_train , losses_3d_valid


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(args):
    device = torch.device(args.device)
    ddp_model = PELTRA(device, bs=args.bs)
    print("INFO: Using PEBRT and Gram-Schmidt process to recover SO(3) rotation matrix")
    print("INFO: Model loaded on {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    print("INFO: Training using dataset {}".format(args.dataset))
    args.distributed = True

    if args.distributed:
        print("INFO: Running on DDP")
        local_rank = args.local_rank
        random_seed = args.random_seed
        set_random_seeds(random_seed=random_seed)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda:{}".format(local_rank))
        model = PELTRA(device, bs=args.bs)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    model_params = 0
    for parameter in ddp_model.parameters():
        model_params += parameter.numel()
    print("INFO: Trainable parameter count:", model_params, " (%.2f M)" %(model_params/1e06))

    train_dataset = Data(args.dataset, transforms)
    val_dataset = Data(args.dataset, transforms, train=False)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.bs, \
            num_workers=args.num_workers, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, \
                num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, \
            num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, \
            num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    if args.resume:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        checkpoint = torch.load(args.resume, map_location=map_location)
        ddp_model.load_state_dict(checkpoint["model"])

        if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

        print("INFO: Using optimizer {}".format(optimizer))

    train_list, val_list = train(args.start_epoch, args.epoch, 
                                train_loader, val_loader, ddp_model, device,
                                optimizer, lr_scheduler, local_rank)

if __name__ == "__main__":
    main(args)
