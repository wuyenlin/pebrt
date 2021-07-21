#!/usr/bin/python3
from common.options import args_parser
from common.petr import *
from common.dataloader import *
from common.loss import *
from common.human import *
from common.misc import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


def train(start_epoch, epoch, train_loader, val_loader, model, device, optimizer, lr_scheduler):
    print("Training starts...")

    losses_3d_train = []
    losses_3d_valid = []

    for ep in tqdm(range(start_epoch, epoch)):
        start_time = time()
        epoch_loss_3d_train = 0.0
        N = 0

        if ep%5 == 0 and ep != 0:
            exp_name = "./petr/all_2_lay_epoch_{}_h36m.bin".format(ep)
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
            _, images, _, inputs_3d = data
            inputs_3d = inputs_3d.to(device)
            images = images.to(device)

            optimizer.zero_grad()

            predicted_3d_pos = model(images)

            loss_3d_pos = anth_mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * loss_3d_pos.item()
            N += inputs_3d.shape[0]

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
                _, images, _, inputs_3d = data
                inputs_3d = inputs_3d.to(device)
                images = images.to(device)

                predicted_3d_pos = model(images)

                loss_3d_pos = anth_mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_valid += inputs_3d.shape[0] * loss_3d_pos.item()
                N += inputs_3d.shape[0]

            losses_3d_valid.append(epoch_loss_3d_valid / N)

        lr_scheduler.step()
        elapsed = (time() - start_time)/60

        print("[%d] time %.2f 3d_train %f 3d_valid %f" % (
                ep + 1,
                elapsed,
                losses_3d_train[-1] * 1000,
                losses_3d_valid[-1] * 1000))

        if args.export_training_curves and ep > 3:
            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], "--", color="C0")
            plt.plot(epoch_x, losses_3d_valid[3:], color="C1")
            plt.legend(["3d train", "3d valid (eval)"])
            plt.ylabel("MPJPE (m)")
            plt.xlabel("Epoch")
            plt.xlim((3, epoch))
            plt.savefig("./checkpoint/loss_3d.png")

            plt.close("all")

    print("Finished Training.")
    return losses_3d_train , losses_3d_valid


def evaluate(test_loader, model, device):
    epoch_loss_3d_pos = 0.0
    epoch_loss_3d_n2 = 0

    with torch.no_grad():
        model.eval()
        N = 0
        for data in test_loader:
            _, images, _, inputs_3d = data
            inputs_3d = inputs_3d.to(device)
            images = images.to(device)

            predicted_3d_pos = model(images)

            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos += inputs_3d.shape[0] * error.item()
            N += inputs_3d.shape[0]


            # convert bone kpts (17,3) to rotation matrix (16,9)
            h = Human(1.8, "cpu")
            model = h.update_pose()
            t_info = vectorize(model)[:,:3]
            pred = torch.zeros(predicted_3d_pos.shape[0], 16, 9)
            tar = torch.zeros(inputs_3d.shape[0], 16, 9)
            for pose in range(predicted_3d_pos.shape[0]):
                pred[pose,:,:] = torch.from_numpy(convert_gt(predicted_3d_pos[pose,:,:], t_info, dataset="h36m"))
                tar[pose,:,:] = torch.from_numpy(convert_gt(inputs_3d[pose,:,:], t_info, dataset="h36m"))
                
            # new metrics
            n2 = mbve(pred, tar)
            epoch_loss_3d_n2 += inputs_3d.shape[0] * n2.item()


    e1 = (epoch_loss_3d_pos / N)*1000
    n2 = (epoch_loss_3d_n2 / N)*1000

    print("Protocol #1 Error (MPJPE):", e1, "mm")
    print("New Metric #2 Error (MPBVE):", n2, "mm")
    print("----------")

    return e1, n2


def run_evaluation(actions, model):
    error_e1 = []
    errors_n2 = []
    for action in actions:
        test_dataset = Data(args.dataset, transforms, False, action)
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=args.num_workers, collate_fn=collate_fn)
        print("-----"+action+"-----")
        e1, n2 = evaluate(test_loader, model, args.device)
        error_e1.append(e1)
        errors_n2.append(n2)
        print()
    print("Protocol #1   (MPJPE) action-wise average:", round(np.mean(error_e1), 1), "mm")
    print("New Metric #2   (MPBVE) action-wise average:", round(np.mean(errors_n2), 1), "mm")


def set_random_seeds(random_seed=0):
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(args):
    device = torch.device(args.device)
    ddp_model = PETR(device, num_layers=args.num_layers)
    ddp_model = ddp_model.to(device)
    print("INFO: Model loaded on {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    print("INFO: Training using dataset {}".format(args.dataset))

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        print("INFO: Running on SLI")
        local_rank = args.local_rank
        random_seed = args.random_seed
        set_random_seeds(random_seed=random_seed)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda:{}".format(local_rank))
        model = PETR(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    backbone_params = 0
    if args.lr_backbone == 0:
        print("INFO: Freezing HRNet")
        for param in ddp_model.backbone.parameters():
            param.requires_grad = False
            backbone_params += param.numel()

    model_params = 0
    for parameter in ddp_model.parameters():
        model_params += parameter.numel()
    if args.lr_backbone == 0:
        model_params -= backbone_params

    print("INFO: Trainable parameter count:", model_params, " (%.2f M)" %(model_params/1e06))

    if args.eval:
        actions = ["Directions", "Discussion", "Eating", "Greeting", "Phoning",
                "Photo",  "Posing", "Purchases", "Sitting", "SittingDown", 
                "Smoking", "Waiting", "Walking", "WalkDog", "WalkTogether"]
        checkpoint = torch.load(args.resume, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model"])
        print("Evaluation starts...")
        run_evaluation(actions, ddp_model)

    else:
        train_dataset = Data(args.dataset, transforms)
        val_dataset = Data(args.dataset, transforms, False)
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


        param_dicts = [
            {"params": [p for n, p in ddp_model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in ddp_model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

        optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

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
                                    optimizer, lr_scheduler)


if __name__ == "__main__":
    args = args_parser()
    main(args)
