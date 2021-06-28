#!/usr/bin/python3
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from common.petr import PETR
from common.pebrt import PEBRT
from common.human import *


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 


def att():
    model = PETR(device="cuda:0",lift=True)
    model.load_state_dict(torch.load('./checkpoint/ft_5.bin')['model'])
    model = model.cuda()
    model.eval()

    img = "dataset/S5/Seq2/imageSequence/video_7/frame004884.jpg"
    img_read = Image.open(img)
    img = transforms(img_read)
    img = img.unsqueeze(0)
    img = img.cuda()
    output = model(img)


def extract_bone(pred, bone, k):
    out = (pred[:,bone[0],k].tolist()[0], pred[:,bone[1],k].tolist()[0])
    return out


def plot3d(ax, bones, output):
    ax.scatter(output[:,:,0], output[:,:,1], output[:,:,2])
    for bone in bones:
        xS = extract_bone(output, bone, 0)
        yS = extract_bone(output, bone, 1)
        zS = extract_bone(output, bone, 2)
        ax.plot(xS, yS, zS)
    # ax.view_init(elev=-80, azim=-90)
    # ax.autoscale()
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    # ax.set_zlim(-1,1)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.view_init(elev=5, azim=90)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel("Y")
    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel("Z")


def plot_human(ax, bones, output):
    for p in output:
        ax.scatter(p[0], p[1], p[2], c='r', alpha=0.5)

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


def viz(bones, img_list, compare=False, savefig=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PETR(device)
    # model.load_state_dict(torch.load('./checkpoint/ft_5.bin')['model'])
    # model.load_state_dict(torch.load('./petr/ft_1_h36m.bin')['model'])
    model.load_state_dict(torch.load('./petr/epoch_45_h36m.bin')['model'])
    model = model.cuda()
    model.eval()
    if compare:
        model_2 = PEBRT(device)
        model_2.load_state_dict(torch.load('./pebrt/epoch_25.bin')['model'])
        model_2 = model_2.cuda()
        model_2.eval()

    fig = plt.figure()
    num_row = 3 if comp else 2

    for k in range(len(img_list)):
        img_read = Image.open(img_list[k])
        img = transforms(img_read)
        img = img.unsqueeze(0)
        img = img.cuda()

# 1st row
        ax = fig.add_subplot(num_row, len(img_list), k+1)
        ax.imshow(img_read)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

# 2nd row
        output = model(img)
        output = output.cpu().detach().numpy()
        ax = fig.add_subplot(num_row, len(img_list), k+len(img_list)+1, projection='3d')
        plot3d(ax, bones, output)

# 3rd row
        if compare:
            h = Human(1.8, "cpu")
            output = model_2(img)
            # output = h.update_pose(output.detach().numpy())
            output = h.update_pose(output)
            ax = fig.add_subplot(num_row, len(img_list), k+2*len(img_list), projection='3d')
            plot_human(ax, bones, output)
            plt.xlim(-1,1)
            plt.ylim(-1,1)

    plt.show()
    if savefig:
        plt.savefig('./checkpoint/this.svg', format='svg', dpi=1200)


if __name__ == "__main__":
    imgs = [
        #0
            ["dataset/S1/Seq1/imageSequence/video_4/frame001049.jpg",
            "dataset/S1/Seq1/imageSequence/video_5/frame001182.jpg",
            "dataset/S1/Seq1/imageSequence/video_8/frame000049.jpg",
            "dataset/S1/Seq1/imageSequence/video_2/frame002453.jpg"],

        #1
            ["dataset/S1/Seq1/imageSequence/video_1/frame000684.jpg",
            "dataset/S1/Seq1/imageSequence/video_1/frame000232.jpg",
            "dataset/S1/Seq2/imageSequence/video_8/frame010424.jpg",
            "dataset/S1/Seq2/imageSequence/video_5/frame008665.jpg"],

        #2
            ["dataset/S4/Seq1/imageSequence/video_5/frame000283.jpg",
            # "dataset/S4/Seq1/imageSequence/video_7/frame002168.jpg",
            "dataset/S4/Seq1/imageSequence/video_7/frame004447.jpg",
            "dataset/S7/Seq1/imageSequence/video_8/frame001549.jpg",
            "dataset/S8/Seq1/imageSequence/video_0/frame005071.jpg"],

        #3
            ["dataset/S5/Seq2/imageSequence/video_7/frame004884.jpg",
            "dataset/S6/Seq2/imageSequence/video_6/frame002665.jpg",
            "dataset/S7/Seq2/imageSequence/video_4/frame001103.jpg",
            "dataset/S8/Seq2/imageSequence/video_1/frame001069.jpg"],

        #4
            ["dataset/S5/Seq1/imageSequence/video_1/frame002604.jpg",
            "dataset/S5/Seq2/imageSequence/video_8/frame012060.jpg",
            "dataset/S6/Seq1/imageSequence/video_7/frame003115.jpg",
            "dataset/S6/Seq1/imageSequence/video_8/frame005868.jpg"],

        #5 bad
            ["dataset/S1/Seq2/imageSequence/video_5/frame012054.jpg",
            "dataset/S2/Seq1/imageSequence/video_5/frame004584.jpg",
            "dataset/S3/Seq2/imageSequence/video_4/frame001103.jpg",
            "dataset/S4/Seq2/imageSequence/video_4/frame002144.jpg"],

        #6
            ["./h36m/S1/Phoning 1.54138969/frame000385.jpg",
            "./h36m/S1/Waiting 1.54138969/frame001220.jpg",
            "./h36m/S1/Walking.54138969/frame001095.jpg",
            "./h36m/S1/Photo.54138969/frame000663.jpg"],

        #7
            ["./h36m/S1/Greeting 1.54138969/frame000376.jpg",
            "./h36m/S1/Eating.54138969/frame002624.jpg",
            "./h36m/S1/Discussion.54138969/frame000673.jpg",
            "./h36m/S1/WalkTogether 1.54138969/frame000156.jpg"],

        #8
            ["./h36m/S1/WalkTogether.54138969/frame000697.jpg",
            "./h36m/S1/Walking.54138969/frame001092.jpg",
            "./h36m/S1/Purchases.54138969/frame000437.jpg",
            "./h36m/S1/SittingDown.54138969/frame001702.jpg"],

        #9
            ["./h36m/S7/Directions 1.54138969/frame000655.jpg",
            "./h36m/S9/Directions.54138969/frame002191.jpg",
            "./h36m/S5/Directions 1.54138969/frame004797.jpg",
            "./h36m/S6/Directions 1.54138969/frame000965.jpg"],

        ]

    
    bones = {
        "mpi": (
            (2,1), (1,0), (0,3), (3,4),  # spine + head
            (0,5), (5,6), (6,7), 
            (0,8), (8,9), (9,10), # arms
            (2,14), (11,12), (12,13),
            (2,11), (14,15), (15,16) # legs
        ),
        "h36m": (
            (0,7), (7,8), (8,9), (9,10),  # spine + head
            (8,14), (14,15), (15,16), 
            (8,11), (11,12), (12,13), # arms
            (0,1), (1,2), (2,3),
            (0,4), (4,5), (5,6) # legs
        )
    }
    comp = False
    viz(bones["h36m"], imgs[8], comp)
