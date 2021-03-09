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
            for vid in [0,1,2,4,5,6]:
                for frame in data[vid].keys():
                    self.kpts.append(data[vid][frame]['keypoints'])
                    self.img_path.append(data[vid][frame]['directory'])
                
        else:
            for vid in [7,8]:
                for frame in data[vid].keys():
                    self.kpts.append(data[vid][frame]['keypoints'])
                    self.img_path.append(data[vid][frame]['directory'])

        self.transforms = transforms

    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            kpts = self.kpts[index].reshape(-1,3)
        except:
            return None
        return img_path, img, kpts

    def __len__(self):
        return len(self.kpts)
    
    def pop_joints(self):
        pass

if __name__ == "__main__":

    train_csv = "dataset/S1/Seq1/imageSequence/S1seq1.npz"
    train_dataset = Data(train_csv, transforms)
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16, drop_last=False)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(labels)
    
    # pts = labels[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    # plt.show()