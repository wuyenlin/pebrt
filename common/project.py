import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image, ImageEnhance, ImageFilter


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
        self.c2i= []
        self.w2i= []
        self.transforms = transforms

        if train:
            vid_list = np.arange(6)
        else:
            vid_list = np.arange(6,8)

        for vid in vid_list:
            for frame in data[vid].keys():
                pts_2d = (data[vid][frame]['2d_keypoints']).reshape(-1,2)
                pro_pts_2d = self.zero_center(self.pop_joints(pts_2d))
                self.gt_pts2d.append(torch.from_numpy(pro_pts_2d))

                pts_3d = (data[vid][frame]['3d_keypoints']).reshape(-1,3)
                cam_3d = self.to_camera_coordinate(pts_2d, pts_3d, vid)
                gt_3d = self.zero_center(cam_3d)/1000
                self.c2i.append(self.cam_to_img(gt_3d, vid))
                self.w2i.append(self.project(pts_2d, pts_3d, vid))
                self.img_path.append(data[vid][frame]['directory'])

    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            gt_2d = self.gt_pts2d[index]
            c2i = self.c2i[index]
            w2i = self.w2i[index]
        except:
            return None
        return img_path, img, gt_2d, c2i, w2i

    def __len__(self):
        return len(self.img_path)
    

    def pop_joints(self, kpts):
        """
        Get 17 joints from the original 28 
        :param kpts: orginal kpts from MPI-INF-3DHP (an array of (28,3))
        :return new_skel: an array of (17,3)
        """
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
        :param camera:              camera number (used in MPI dataset)
        :return intrinsic matrix:
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
        self.R, self.t = R, t
        return cam_3d

    
    def zero_center(self, cam):
        """
        translate root joint to origin (0,0,0)
        """
        return cam - cam[2,:]


    def project(self, pts_2d, pts_3d, camera):
        """
        Project 3D kpts (in World coordinates) to 2D Image coordinates
        """
        self.to_camera_coordinate(pts_2d, pts_3d, camera)
        proj, _ = cv.projectPoints(pts_3d, self.R, self.t, self.intrinsic, np.zeros(4))
        return self.pop_joints(proj)


    def cam_to_img(self, gt_3d, camera):
        """
        Project processed 3D kpts (in Camera coordinates) to 2D Image coordinates
        """
        self.get_intrinsic(camera)
        img_coor = self.intrinsic @ gt_3d.transpose()
        return img_coor.transpose()



def verify_proj(transform):
    """
    This function verifies that the 3D pose can be projected back to 2D img
    """
    train_npz = "dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transform, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=2, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, gt_2d, c2i, w2i = dataiter.next()
    
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    fig = plt.figure()
    ax = fig.add_subplot(141)
    img = Image.open(img_path[0])
    # plt.imshow(img.resize((256,256)))
    plt.imshow(img)

    pts = gt_2d[0]
    ax = fig.add_subplot(142)
    ax.scatter(pts[:,0], pts[:,1])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        
        ax.plot(xS, yS)
    ax.invert_yaxis()

    # 2nd - w2i
    pts = w2i[0]
    ax = fig.add_subplot(143)
    ax.scatter(pts[:,0], pts[:,1])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        
        ax.plot(xS, yS)
    ax.invert_yaxis()
    
    # 3rd - c2i
    pts = c2i[0]
    ax = fig.add_subplot(144)
    ax.scatter(pts[:,0], pts[:,1])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        
        ax.plot(xS, yS)
    ax.invert_yaxis()

    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    verify_proj(transform)