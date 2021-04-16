import os
import cv2 as cv
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter


def imshow(img):
    img = img / 2 + 0.5   
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Data:
    def __init__(self, npz_path, transforms = None, train=True):
        data = np.load(npz_path, allow_pickle=True)
        data = data["arr_0"].reshape(1,-1)[0]

        self.img_path = []
        self.gt_pts2d = []
        self.gt_pts3d = []
        self.transforms = transforms

        if train:
            vid_list = np.arange(6)
        else:
            vid_list = np.arange(6,8)

        for vid in vid_list:
            for frame in data[vid].keys():
                pts_2d = (data[vid][frame]['2d_keypoints']).reshape(-1,2)
                pro_pts_2d = self.zero_center(self.pop_joints(pts_2d)*96/2048)
                self.gt_pts2d.append(torch.from_numpy(pro_pts_2d))

                pts_3d = (data[vid][frame]['3d_keypoints']).reshape(-1,3)
                cam_3d = self.to_camera_coordinate(pts_2d, pts_3d, vid)
                self.gt_pts3d.append(torch.from_numpy(self.zero_center(cam_3d)/1000))
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
    

    def pop_joints(self, kpts):
        '''
        Get 17 joints from the original 28 
        '''
        new_skel = np.zeros([17,3]) if kpts.shape[-1]==3 else np.zeros([17,2])
        ext_list = [0,2,4,5,6,         # spine+head
                    9,10,11,14,15,16,  # arms
                    18,19,20,23,24,25] # legs
        for row in range(17):
            new_skel[row, :] = kpts[ext_list[row], :]
        return new_skel


    def get_intrinsic(self, camera):
        """
        Parse camera matrix from calibration file
        """
        calib = open("./dataset/S1/Seq1/camera.calibration","r")
        content = calib.readlines()
        content = [line.strip() for line in content]
        # 3x3 intrinsic matrix
        intrinsic = np.array(content[7*camera+5].split(" ")[3:], dtype=np.float32)
        intrinsic = np.reshape(intrinsic, (4,-1))
        self.intrinsic = intrinsic[:3, :3]


    def to_camera_coordinate(self, pts_2d, pts_3d, camera):
        self.get_intrinsic(camera)
        ret, R, t= cv.solvePnP(pts_3d, pts_2d, self.intrinsic, np.zeros(4), flags=cv.SOLVEPNP_EPNP)

        # get extrinsic matrix
        assert ret
        R = cv.Rodrigues(R)[0]
        E = np.concatenate((R,t), axis=1)  # [R|t], a 3x4 matrix
    
        pts_3d = cv.convertPointsToHomogeneous(self.pop_joints(pts_3d)).transpose().squeeze(1)
        cam_coor = E @ pts_3d
        cam_3d = cam_coor.transpose()
        return cam_3d

    
    def zero_center(self, cam):
        return cam - cam[2,:]



def test():
    train_npz = "dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transforms, True)
    print(len(train_dataset))
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, kpts, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    pts = labels[0]
    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.imshow(Image.open(img_path[0]))
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        zS = (pts[bone[0],2], pts[bone[1],2])
        
        ax.plot(xS, yS, zS)
    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def exp():
    train_npz = "dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, kpts, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    pts = labels[0]
    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.imshow(Image.open(img_path[0]))
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        zS = (pts[bone[0],2], pts[bone[1],2])
        
        ax.plot(xS, yS, zS)
    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":

    transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    # test()
    rmat_z = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    rmat_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    rot = rmat_z @ rmat_x
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rot)
    print(rot)
    print("\n")
    print(angles)
    print(Qx)
    print(Qy)
    print(Qz)