#!/usr/bin/python3
import sys
sys.path.append('../')
import os
import torch
from torchvision import transforms
from common.dataloader import *
from torch.utils.data import DataLoader
from common.pebrt import PEBRT
from common.human import *
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import argparse

parser = argparse.ArgumentParser("Set PEBRT parameters for visualization", add_help=False)

parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--action", type=str, default="Smoking")
parser.add_argument("--dataset", type=str, default="../h36m/data_h36m_frame_all.npz")
parser.add_argument("--device", default="cuda", help="device used")
parser.add_argument("--checkpoint", help="path to pre-trained weights")

args = parser.parse_args()

transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 

bones = (
    (2,1), (1,0), (0,3), (3,4),  # spine + head
    (0,5), (5,6), (6,7), 
    (0,8), (8,9), (9,10), # arms
    (2,11), (11,12), (12,13),
    (2,14), (14,15), (15,16), # legs
    )


def get_frame(path, file_list, k):
    img_path = "." + path + file_list[k]
    return Image.open(img_path)


def viz(args):
    print("Loading data")

    train_dataset = Data(args.dataset, transforms, train=False, action=args.action)
    trainloader = DataLoader(train_dataset, batch_size=args.bs, \
                        shuffle=False, num_workers=8, drop_last=True)
    print("Data loaded!")
    dataiter = iter(trainloader)
    img_path, kpts, _, _ = dataiter.next()
    print(img_path)
    path = img_path[0].split("frame")[0]
    print(path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PEBRT(device, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    print("INFO: Loaded checkpoint from ", args.checkpoint)
    model = model.to(device)
    model.eval()

    stack = {}
    for k in tqdm(range(1,args.bs+1)):
        output, _ = model(kpts[k-1].unsqueeze(0).to(device))
        stack[k-1] = output.detach().cpu().numpy()

    np.savez_compressed("pose_stack", stack)
    print("INFO: npz file saved. \n")

    return path


def animate(args, bones, format="mp4"):
    path = viz(args)

    fig = plt.figure()
    # animate dataset image stream
    ax1 = fig.add_subplot(121)
    file_list = sorted(os.listdir("."+path))
    im = ax1.imshow(get_frame(path, file_list, 0))

    # animate 3D pose
    data = np.load("./pose_stack.npz", allow_pickle=True)
    data = data["arr_0"].reshape(1,-1)[0][0]
    ax2 = fig.add_subplot(122, projection='3d')
    # Setting the axes properties
    ax2.set_xlim3d([-1.0, 1.0])
    ax2.set_xticklabels([])

    ax2.set_ylim3d([-1.0, 1.0])
    ax2.set_yticklabels([])

    ax2.set_zlim3d([-1.0, 1.0])
    ax2.set_zticklabels([])

    ax2.set_title('Reconstruction')
    ax2.view_init(elev=20, azim=80)

    h = Human(1.7, "cpu")
    output = h.update_pose(data[0])
    output = output.detach().numpy()

    # Initialize scatters
    scatters = [ ax2.scatter(output[p,0:1], output[p,1:2], output[p,2:], c='r') for p in range(output.shape[0]) ]

    # Initialize lines
    lines_3d = [[] for _ in range(len(bones))]
    for n, bone in enumerate(bones):
        xS = (output[bone[0]][0],output[bone[1]][0])
        yS = (output[bone[0]][1],output[bone[1]][1])
        zS = (output[bone[0]][2],output[bone[1]][2])
        lines_3d[n].append(ax2.plot(xS, yS, zS, linewidth=5))


    def update(iter, data, bones):
        im.set_data(get_frame(path, file_list, iter))
        h = Human(1.7, "cpu")
        out_pose = h.update_pose(data[iter])
        out_pose = out_pose.detach().numpy()
        for i in range(out_pose.shape[0]):
            scatters[i]._offsets3d = (out_pose[i,0:1], out_pose[i,1:2], out_pose[i,2:])

        for n, bone in enumerate(bones):
            lines_3d[n][0][0].set_xdata(np.array([out_pose[bone[0]][0],out_pose[bone[1]][0]]))
            lines_3d[n][0][0].set_ydata(np.array([out_pose[bone[0]][1],out_pose[bone[1]][1]]))
            lines_3d[n][0][0].set_3d_properties(np.array([out_pose[bone[0]][2],out_pose[bone[1]][2]]), zdir="z")


    # Number of iterations
    iterations = len(data)
    print("number of frames:", iterations)
    print("Processing...")

    anim = FuncAnimation(fig, update, iterations, fargs=(data, bones), \
                        interval=100, blit=False, repeat=False)

    if format == "mp4":
        Writer = writers['ffmpeg']
        writer = Writer(fps=10, metadata={})
        anim.save("output.mp4", writer=writer)
    elif format == "gif":
        anim.save("output.gif", dpi=80, writer='imagemagick')
    else:
        print("Unsupported file format")

    plt.close()


if __name__ == "__main__":
    animate(args, bones)