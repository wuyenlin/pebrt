#!/usr/bin/python3
import numpy as np
import torch
from torchvision import transforms
from common.dataloader import *
from torch.utils.data import DataLoader
from common.peltra import PELTRA
from common.human import *
from tqdm import tqdm



transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 


bones = (
    (2,1), (1,0), (0,3), (3,4),  # spine + head
    (0,5), (5,6), (6,7), 
    (0,8), (8,9), (9,10), # arms
    (2,14), (11,12), (12,13),
    (2,11), (14,15), (15,16), # legs
)


def plot3d(ax, bones, output):
    for p in output:
        ax.scatter(p[0], p[1], p[2], c='r', alpha=0.5)

    for index in bones:
        xS = (output[index[0]][0],output[index[1]][0])
        yS = (output[index[0]][1],output[index[1]][1])
        zS = (output[index[0]][2],output[index[1]][2])
        ax.plot(xS, yS, zS)
    ax.view_init(elev=-80, azim=-90)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel("Y")
    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel("Z")


def viz():
    print("Loading data")
    train_npz = "./dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=512, 
                        shuffle=False, num_workers=8, drop_last=True)
    print("Data loaded!")
    dataiter = iter(trainloader)
    img_path, image, kpts, labels = dataiter.next()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PELTRA(device)
    net.load_state_dict(torch.load('./peltra/ft_1_zero.bin')['model'])
    net = net.cuda()
    net.eval()

    data = {}
    for k in tqdm(range(1,513)):
        pts = torch.tensor(kpts[k-1].unsqueeze(0)).cuda()
        output = net(pts)
        h = Human(1.8, "cpu")
        output = h.update_pose(output.detach().numpy())
        data[k-1] = output.detach().numpy()

    np.savez_compressed("pose_seq", data)


if __name__ == "__main__":
    viz()