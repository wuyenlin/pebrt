#!/usr/bin/python3

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from common.dataloader import *


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 

def superimpose(bones):
    train_npz = "dataset/S1/Seq1/imageSequence/S1Seq1.npz"
    train_dataset = Data(train_npz, transforms, False)
    print("Loading data")
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True)
    print("Data loaded!")
    dataiter = iter(trainloader)
    img_path, images, kpts, vec = dataiter.next()

    fig = plt.figure()
    img_read = Image.open(img_path[0])

    ax = fig.add_subplot(111)
    ax.imshow(img_read)
    pts = kpts[0]
    ax.scatter(pts[:,0], pts[:,1])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        
        ax.plot(xS, yS)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


if __name__ == "__main__":
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )
    superimpose(bones)