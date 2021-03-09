import os
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

def pop_joints(kpts):
    '''
    Get 17 joints from the original 28 
    '''
    new_skel = np.zeros([17,3])
    ext_list = [0,2,4,6,7,9,10,11,14,15,16,
                18,19,20,23,24,25]
    for row in range(17):
        new_skel[row, :] = kpts[ext_list[row], :]
    return new_skel

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.CenterCrop(180),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]) 

def imshow(img):
    img = img / 2 + 0.5   
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class Data:
    def __init__(self, npz_path, transforms = None, train=True):
        data = np.load(npz_path, allow_pickle=True)
        data = data["arr_0"].reshape(1,-1)[0]

        self.img_path = []
        self.kpts = []

        if train:
            '''
            For training purpose
            '''
            for vid in range(6):
                for frame in data[vid].keys():
                    pts = data[vid][frame]['keypoints'].reshape(28,3)
                    self.kpts.append(pop_joints(pts))
                    self.img_path.append(data[vid][frame]['directory'])
                
        else:
            '''
            For testing purpose
            '''
            for vid in range(6,8):
                for frame in data[vid].keys():
                    pts = data[vid][frame]['keypoints'].reshape(28,3)
                    self.kpts.append(pop_joints(pts))
                    self.img_path.append(data[vid][frame]['directory'])

        self.transforms = transforms

    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            kpts = self.kpts[index]
        except:
            return None
        return img_path, img, kpts

    def __len__(self):
        return len(self.kpts)
    

if __name__ == "__main__":

    train_npz = "dataset/S1/Seq1/imageSequence/S1seq1.npz"
    train_dataset = Data(train_npz, transforms, False)
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16, drop_last=False)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    
    pts = labels[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    plt.show()