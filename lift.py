#!/usr/bin/python3

from common.peltra import *
from common.dataloader import *
from common.loss import *
from common.human import *

import argparse
from tqdm import tqdm
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time

parser = argparse.ArgumentParser("Set PEBRT parameters", add_help=False)

# Hyperparameters
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--bs", type=int, default=2)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-04)
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
parser.add_argument("--local_rank", type=int, help="local rank")
parser.add_argument("--random_seed", type=int, help="random seed", default=0)

args = parser.parse_args()


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


def train(start_epoch, epoch, train_loader, val_loader, model, device, optimizer, lr_scheduler):
    print("Training starts...")

    losses_3d_train = []
    losses_3d_valid = []

    for ep in tqdm(range(start_epoch, epoch+1)):
        start_time = time()
        epoch_loss_3d_train = 0.0
        N = 0
        if ep%5 == 0 and ep != 0:
            exp_name = "./peltra/new_4_lay_epoch_{}.bin".format(ep)
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
            _, inputs_2d, inputs_3d, vec_3d = data
            inputs_2d = inputs_2d.to(device)
            inputs_3d = inputs_3d.to(device)
            vec_3d = vec_3d.to(device)

            optimizer.zero_grad()

            predicted_3d, w_kc = model(inputs_2d)

            loss_3d_pos = maev(predicted_3d, vec_3d, w_kc) 
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
                _, inputs_2d, inputs_3d, vec_3d = data
                inputs_2d = inputs_2d.to(device)
                inputs_3d = inputs_3d.to(device)
                vec_3d = vec_3d.to(device)

                predicted_3d, w_kc = model(inputs_2d)

                loss_3d_pos = maev(predicted_3d, vec_3d, w_kc)
                epoch_loss_3d_valid += vec_3d.shape[0] * loss_3d_pos.item()
                N += vec_3d.shape[0]

            losses_3d_valid.append(epoch_loss_3d_valid / N)

        lr_scheduler.step()
        elapsed = (time() - start_time)/60

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


def evaluate(test_loader, model, device):
    epoch_loss_e0 = 0.0
    epoch_loss_n1 = 0.0
    epoch_loss_n2 = 0.0

    with torch.no_grad():
        N = 0
        for data in test_loader:
            _, inputs_2d, inputs_3d, vec_3d = data
            inputs_2d = inputs_2d.to(device)
            inputs_3d = inputs_3d.to(device)
            vec_3d = vec_3d.to(device)

            predicted_3d_pos, _ = model(inputs_2d)

            pose_stack = torch.zeros(predicted_3d_pos.size(0),17,3)
            for b in range(predicted_3d_pos.size(0)):
                h = Human(1.8, "cpu")
                pose_stack[b] = h.update_pose(predicted_3d_pos[b].detach().cpu().numpy())
            e0 = mpjpe(pose_stack, inputs_3d)
            n1 = maev(predicted_3d_pos, vec_3d)
            n2 = mpbve(predicted_3d_pos, vec_3d, 0)
            
            epoch_loss_e0 += vec_3d.shape[0] * e0.item()
            epoch_loss_n1 += vec_3d.shape[0] * n1.item()
            epoch_loss_n2 += vec_3d.shape[0] * n2.item()
            N += vec_3d.shape[0]

            e0 = (epoch_loss_e0 / N)*1000
            n1 = epoch_loss_n1 / N
            n2 = (epoch_loss_n2 / N)*1000

    print("Protocol #0 Error (MPJPE):\t", e0, "\t(mm)")
    print("New Metric #1 Error (MAEV):\t", n1)
    print("New Metric #2 Error (MPBVE):\t", n2, "\t(mm)")
    print("----------")
    
    return e0, n1, n2


def run_evaluation(model, actions=None):
    """ Evalution on Human3.6M dataset """
    error_e0 = []
    errors_n1 = []
    errors_n2 = []
    if actions is not None:
        # evaluting on h36m
        model.load_state_dict(torch.load("./peltra/new_2_lay_epoch_10.bin")["model"])
        model = model.cuda()
        model.eval()
        for action in actions:
            test_dataset = Data(args.dataset, transforms, False, action)
            test_loader = DataLoader(test_dataset, batch_size=512, drop_last=True, shuffle=False, \
                                    num_workers=args.num_workers, collate_fn=collate_fn)
            print("-----"+action+"-----")
            e0, n1, n2 = evaluate(test_loader, model, args.device)
            error_e0.append(e0)
            errors_n1.append(n1)
            errors_n2.append(n2)
        print("Protocol #1   (MPJPE) action-wise average:", round(np.mean(error_e0), 1), "(mm)")
        print("New Metric #1   (MAEV) action-wise average:", round(np.mean(errors_n1), 1), "-")
        print("New Metric #2   (MPBVE) action-wise average:", round(np.mean(errors_n2), 1), "(mm)")
    else:
        # evaluting on MPI-INF-3DHP
        test_dataset = Data(args.dataset, transforms, False)
        test_loader = DataLoader(test_dataset, batch_size=512, drop_last=True,
                                num_workers=args.num_workers, collate_fn=collate_fn)
        e0, n1, n2 = evaluate(test_loader, model, args.device)


def main(args):
    device = torch.device(args.device)
    model = PELTRA(device, bs=args.bs, num_layers=args.num_layers)
    print("INFO: Using PELTRA and Gram-Schmidt process to recover SO(3) rotation matrix")
    model = model.to(device)
    print("INFO: Model loaded on {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    print("INFO: Training using dataset {}".format(args.dataset))

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print("INFO: Trainable parameter count:", model_params, " (%.2f M)" %(model_params/1e06))

    if args.eval:
        if "h36m" in args.dataset:
            actions = ["Directions", "Discussion", "Eating", "Greeting", "Phoning",
                    "Photo",  "Posing", "Purchases", "Sitting", "SittingDown", 
                    "Smoking", "Waiting", "Walking", "WalkDog", "WalkTogether"]
            print("Evaluation on Human3.6M starts...")
            run_evaluation(model, actions)
        else:
            print("Evaluation on MPI-INF-3DHP starts...")
            run_evaluation(model)

    else:
        train_dataset = Data(args.dataset, transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.bs, \
            shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

        val_dataset = Data(args.dataset, transforms, False)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, \
            shuffle=False, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(checkpoint["model"])

            if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1

        print("INFO: Using optimizer {}".format(optimizer))

        train_list, val_list = train(args.start_epoch, args.epoch, 
                                    train_loader, val_loader, model, device,
                                    optimizer, lr_scheduler)

if __name__ == "__main__":
    main(args)
