#!/usr/bin/python3
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from common.petr import PETR
from common.human import *


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 


def extract_bone(pred, bone, k):
    out = (pred[:,bone[0],k].tolist()[0], pred[:,bone[1],k].tolist()[0])
    return out


def plot3d(ax, bones, output):
    ax.scatter(output[:,:,0], output[:,:,1], output[:,:,2])
    for bone in bones:
        xS = extract_bone(output, bone, 0)
        yS = extract_bone(output, bone, 1)
        zS = extract_bone(output, bone, 2)
        ax.plot(xS, yS, zS, linewidth=5)
    ax.view_init(elev=20, azim=80)

    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel("Y")
    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel("Z")




def viz(bones, img_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PETR(device, num_layers=4)
    model.load_state_dict(torch.load('./petr/all_4_lay_latest_h36m.bin')['model'])
    model = model.cuda()
    model.eval()

    fig = plt.figure(figsize=(60/2.54,30/2.54))

    for k in range(len(img_list)):
        img_read = Image.open(img_list[k])
        img = transforms(img_read)
        img = img.unsqueeze(0)
        img = img.cuda()

# 1st row
        ax = fig.add_subplot(1, 2, k+1)
        ax.imshow(img_read)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

# 2nd row
        output = model(img)
        output = output.cpu().detach().numpy()
        ax = fig.add_subplot(1, 2, k+len(img_list)+1, projection='3d')
        plot3d(ax, bones, output)

    # plt.show()
    plt.tight_layout(pad=0.7, w_pad=0.7, h_pad=1.0)

    # rotate the axes and update
    for angle in range(-90, 90):
        ax.view_init(0, angle)
        plt.draw()
        plt.pause(0.0001)


if __name__ == "__main__":
    imgs = [
            [
            "./h36m/S11/SittingDown 1.54138969/frame000370.jpg",
            ],

        ]

    
    bones = (
            (2,1), (1,0), (0,3), (3,4),  # spine + head
            (0,5), (5,6), (6,7), 
            (0,8), (8,9), (9,10), # arms
            (2,14), (11,12), (12,13),
            (2,11), (14,15), (15,16) # legs
        )

    viz(bones, imgs[0])
