#!/usr/bin/python3
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from common.dataloader import *
from common.peltra import PELTRA
from common.human import *


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 


bones = (
(0,1), (0,3), (1,2), (3,4),  # spine + head
(0,5), (0,8),
(5,6), (6,7), (8,9), (9,10), # arms
(2,14), (2,11),
(11,12), (12,13), (14,15), (15,16), # legs
)

def plot3d(ax, bones, output):
    for p in output:
        ax.scatter(p[0], p[1], p[2], c='r', alpha=0.7)

    for index in bones:
        xS = (output[index[0]][0],output[index[1]][0])
        yS = (output[index[0]][1],output[index[1]][1])
        zS = (output[index[0]][2],output[index[1]][2])
        ax.plot(xS, yS, zS)
    ax.view_init(elev=-80, azim=-90)
    ax.autoscale()
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def viz(savefig=False):
    train_npz = "dataset/S1/Seq1/imageSequence/S1Seq1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, kpt_2d, kpts, labels = dataiter.next()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PELTRA(device)
    net.load_state_dict(torch.load('./angle_checkpoint/epoch_45.bin')['model'])
    net = net.cuda()
    net.eval()


    fig = plt.figure()
    for k in range(1,5):
        ax = fig.add_subplot(2, 4, k)
        plt.imshow(Image.open(img_path[k-1]))

        h = Human(1.8, "cpu")
        pts = kpt_2d[k-1]
        pts = torch.tensor(pts.unsqueeze(0)).cuda()
        output = net(pts)

        output = h.update_pose(output)
        output = output.detach().numpy()

        ax = fig.add_subplot(2, 4, k+4, projection='3d')
        plot3d(ax, bones, output)
        plt.xlim(-1,1)
        plt.ylim(-1,1)

    plt.show()
    if savefig:
        plt.savefig('./checkpoint/this.svg', format='svg', dpi=1200)


if __name__ == "__main__":
    viz()