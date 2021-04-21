#!/usr/bin/python3
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from common.petr import PETR
from common.petra import PETRA


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
    _, output = model(img)


def viz(bones, group):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PETR(device, lift=True)
    # model.load_state_dict(torch.load('./anth_checkpoint/ft_5.bin')['model'])
    model.load_state_dict(torch.load('./checkpoint/ft_5.bin')['model'])
    model = model.cuda()
    model.eval()

    imgs = [
        #0
            ["dataset/S1/Seq1/imageSequence/video_4/frame001049.jpg",
            "dataset/S1/Seq1/imageSequence/video_5/frame001182.jpg",
            "dataset/S1/Seq1/imageSequence/video_8/frame000049.jpg",
            "dataset/S1/Seq1/imageSequence/video_2/frame002453.jpg"],

        #1
            ["dataset/S1/Seq1/imageSequence/video_1/frame000684.jpg",
            "dataset/S1/Seq1/imageSequence/video_1/frame000232.jpg",
            "dataset/S1/Seq2/imageSequence/video_0/frame006958.jpg",
            "dataset/S1/Seq2/imageSequence/video_5/frame008665.jpg"],

        #2
            ["dataset/S4/Seq1/imageSequence/video_5/frame000283.jpg",
            "dataset/S4/Seq1/imageSequence/video_7/frame002168.jpg",
            "dataset/S7/Seq1/imageSequence/video_8/frame001549.jpg",
            "dataset/S8/Seq1/imageSequence/video_0/frame005071.jpg"],

        #3
            ["dataset/S5/Seq2/imageSequence/video_7/frame004884.jpg",
            "dataset/S6/Seq2/imageSequence/video_6/frame002665.jpg",
            "dataset/S7/Seq2/imageSequence/video_4/frame001103.jpg",
            "dataset/S8/Seq2/imageSequence/video_1/frame001069.jpg"], #bad

        #4
            ["dataset/S5/Seq1/imageSequence/video_1/frame002604.jpg",
            "dataset/S5/Seq2/imageSequence/video_8/frame000659.jpg",
            "dataset/S6/Seq1/imageSequence/video_7/frame003115.jpg",
            "dataset/S6/Seq1/imageSequence/video_8/frame005868.jpg"]

            ]
    img_list = imgs[group]
    k = 1
    fig = plt.figure()
    for i in range(len(img_list)):
        img_read = Image.open(img_list[i])
        img = transforms(img_read)
        img = img.unsqueeze(0)
        img = img.cuda()

        ax = fig.add_subplot(2, len(img_list), k)
        ax.imshow(img_read)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        _, output = model(img)
        output = output.cpu().detach().numpy()
        ax = fig.add_subplot(2, len(img_list), k+len(img_list), projection='3d')
        ax.scatter(output[:,:,0], output[:,:,1], output[:,:,2])
        for bone in bones:
            xS = (output[:,bone[0],0].tolist()[0], output[:,bone[1],0].tolist()[0])
            yS = (output[:,bone[0],1].tolist()[0], output[:,bone[1],1].tolist()[0])
            zS = (output[:,bone[0],2].tolist()[0], output[:,bone[1],2].tolist()[0])
            
            ax.plot(xS, yS, zS)
        ax.view_init(elev=-80, azim=-90)
        ax.autoscale()
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        k += 1
    plt.show()
    plt.savefig('./checkpoint/this.svg', format='svg', dpi=1200)


if __name__ == "__main__":
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )
    import sys
    group = int(sys.argv[1])
    viz(bones, group)
