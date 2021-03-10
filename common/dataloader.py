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
    new_skel = np.zeros([17,3]) if kpts.shape[-1]==3 else np.zeros([17,2])
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
        self.gt_pts2d = []
        self.gt_pts3d = []
        self.transforms = transforms

        if train:
            '''
            For training purpose
            '''
            for vid in range(6):
                for frame in data[vid].keys():
                    pts_2d = (data[vid][frame]['2d_keypoints']).reshape(28,2)
                    self.gt_pts2d.append(torch.from_numpy(pop_joints(pts_2d)))
                    pts_3d = (data[vid][frame]['3d_keypoints']/1000).reshape(28,3)
                    self.gt_pts3d.append(torch.from_numpy(pop_joints(pts_3d)))
                    self.img_path.append(data[vid][frame]['directory'])
                
        else:
            '''
            For testing purpose
            '''
            for vid in range(6,8):
                for frame in data[vid].keys():
                    pts_2d = (data[vid][frame]['2d_keypoints']).reshape(28,2)
                    self.gt_pts2d.append(torch.from_numpy(pop_joints(pts_2d)))
                    pts_3d = (data[vid][frame]['3d_keypoints']/1000).reshape(28,3)
                    self.gt_pts3d.append(torch.from_numpy(pop_joints(pts_3d)))
                    self.img_path.append(data[vid][frame]['directory'])


    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            kpts_2d = self.gt_pts2d[index]
            kpts_3d = self.gt_pts3d[index]
        except:
            return None
        return img_path, img, kpts_2d, kpts_3d

    def __len__(self):
        return len(self.img_path)
    

if __name__ == "__main__":

    train_npz = "dataset/S1/Seq1/imageSequence/S1seq1.npz"
    train_dataset = Data(train_npz, transforms, True)
    print(len(train_dataset))
    print(train_dataset[0])
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, labels, _ = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    
    # pts = labels[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    # plt.show()