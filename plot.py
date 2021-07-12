#!/usr/bin/python3
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from common.dataloader import *
from torch.utils.data import DataLoader
from common.peltra import PELTRA
from common.human import *
from common.misc import *



transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 



def plot3d(ax, output):
    for p in output:
        ax.scatter(p[0], p[1], p[2], c="r", alpha=0.5)

    bones = (
        (2,1), (1,0), (0,3), (3,4),  # spine + head
        (0,5), (5,6), (6,7), 
        (0,8), (8,9), (9,10), # arms
        (2,14), (11,12), (12,13),
        (2,11), (14,15), (15,16) # legs
    )
    for index in bones:
        xS = (output[index[0]][0],output[index[1]][0])
        yS = (output[index[0]][1],output[index[1]][1])
        zS = (output[index[0]][2],output[index[1]][2])
        ax.plot(xS, yS, zS)
    # ax.view_init(elev=-90, azim=-90)
    ax.view_init(elev=20, azim=90)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel("Y")
    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel("Z")


def viz(savefig=False):
    # train_npz = "./dataset/S1/Seq1/imageSequence/S1.npz"
    # train_npz = "./h36m/data_h36m_frame_S1.npz"
    train_npz = "./h36m/data_h36m_frame_all.npz"
    train_dataset = Data(train_npz, transforms, False)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True, collate_fn=collate_fn)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, kpts, gt_3d, vec_3d = dataiter.next()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PELTRA(device, num_layers=2)
    net.load_state_dict(torch.load("./peltra/new_2_lay_epoch_50.bin")["model"])
    net = net.cuda()
    net.eval()

    fig = plt.figure()
    for k in range(1,5):
        ax = fig.add_subplot(2, 4, k)
        plt.imshow(Image.open(img_path[k-1]))

# 2nd row - GT
        # output = gt_3d[k-1]
        # ax = fig.add_subplot(3, 4, k+4, projection="3d")
        # plot3d(ax, output)

# 3rd row - pred
        pts = kpts[k-1].unsqueeze(0).cuda()
        output, _ = net(pts)
        h = Human(1.8, "cpu")
        output = h.update_pose(output)

        ax = fig.add_subplot(2, 4, k+4, projection="3d")
        plot3d(ax, output)

    plt.show()

    if savefig:
        plt.savefig("./checkpoint/this.svg", format="svg", dpi=1200)


if __name__ == "__main__":
    viz()
