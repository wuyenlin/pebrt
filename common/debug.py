from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image, ImageEnhance, ImageFilter
try:
    from common.human import *
except ModuleNotFoundError:
    from human import *


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
        # self.gt_pts2d = []
        self.gt_pts3d = []
        self.gt_vecs3d = []
        self.transforms = transforms

        if train:
            vid_list = np.arange(6)
        else:
            vid_list = np.arange(6,8)

        for vid in vid_list:
            for frame in data[vid].keys():
                pts_2d = (data[vid][frame]['2d_keypoints']).reshape(-1,2)
                pro_pts_2d = self.zero_center(self.pop_joints(pts_2d)*96/2048)
                # self.gt_pts2d.append(torch.from_numpy(pro_pts_2d))

                pts_3d = (data[vid][frame]['3d_keypoints']).reshape(-1,3)
                # cam_3d = self.to_camera_coordinate(pts_2d, pts_3d, vid)
                # gt_3d = self.zero_center(cam_3d)/1000

                gt_3d = self.zero_center(self.pop_joints(pts_3d))/1000
                self.gt_pts3d.append(gt_3d)
                self.gt_vecs3d.append((vectorize(gt_3d)))
                self.img_path.append(data[vid][frame]['directory'])

    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            # kpts_2d = self.gt_pts2d[index]
            kpts_3d = self.gt_pts3d[index]
            vecs_3d = self.gt_vecs3d[index]
        except:
            return None
        # return img_path, img, kpts_2d, kpts_3d
        return img_path, img, kpts_3d, vecs_3d

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
        return cam_3d

    
    def zero_center(self, cam):
        """
        translate root joint to origin (0,0,0)
        """
        return cam - cam[2,:]


def try_load():
    train_npz = "dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=2, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, kpts, labels = dataiter.next()
    print(labels[0])
    
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    fig = plt.figure()
    ax = fig.add_subplot(131)
    plt.imshow(Image.open(img_path[0]))

    # 2nd - 3D Pose
    pts = kpts[0]
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        zS = (pts[bone[0],2], pts[bone[1],2])
        
        ax.plot(xS, yS, zS)
    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 3rd - vectorized 
    pts = labels[0]
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for i in range(pts.shape[0]):
        xS = (0, pts[i,0])
        yS = (0, pts[i,1])
        zS = (0, pts[i,2])
        
        ax.plot(xS, yS, zS)
    
    # unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='r', alpha=0.1)

    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def plot_results(predicted, target):
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    fig = plt.figure()

    # 2nd - 3D Pose
    pts = predicted
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for i in range(pts.shape[0]):
        xS = (0, pts[i,0])
        yS = (0, pts[i,1])
        zS = (0, pts[i,2])
        
        ax.plot(xS, yS, zS)
    
    # unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='r', alpha=0.1)

    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

####
    pts = target
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for i in range(pts.shape[0]):
        xS = (0, pts[i,0])
        yS = (0, pts[i,1])
        zS = (0, pts[i,2])
        
        ax.plot(xS, yS, zS)
    
    # unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='r', alpha=0.1)

    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
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
    # try_load()
    predicted = torch.tensor([[ 0.0868,  0.1927, -0.9774,  0.2448],
         [ 0.9071, -0.4208,  0.0000,  0.0884],
         [ 0.3111, -0.1885,  0.9315,  0.2448],
         [ 0.0020, -0.0030, -1.0000,  0.1122],
         [ 1.0000,  0.0000,  0.0000,  0.2193],
         [-1.0000,  0.0000,  0.0000,  0.2193],
         [ 0.0232,  0.0487, -0.9985,  0.3162],
         [-0.0514,  0.0165, -0.9985,  0.2482],
         [-0.1150, -0.6964,  0.7084,  0.3162],
         [ 0.4904, -0.5076,  0.7084,  0.2482],
         [-1.0000,  0.0000,  0.0000,  0.1623],
         [ 1.0000,  0.0000,  0.0000,  0.1623],
         [ 0.3013,  0.0204,  0.9533,  0.4165],
         [ 0.9873,  0.0668, -0.1440,  0.4182],
         [ 0.1356,  0.0011,  0.9908,  0.4165],
         [ 0.7493,  0.0061, -0.6622,  0.4182]])

        
    target = torch.tensor([[ 0.0502,  0.9907, -0.1267,  0.0791],
         [ 0.0984, -0.9880, -0.1189,  0.1794],
         [ 0.0685,  0.9960,  0.0565,  0.2274],
         [ 0.0629, -0.9935, -0.0950,  0.0865],
         [ 0.8325, -0.5326, -0.1526,  0.1821],
         [-0.5550, -0.8300, -0.0551,  0.1921],
         [ 0.0463,  0.8915, -0.4507,  0.3170],
         [-0.4699,  0.8728, -0.1318,  0.2450],
         [-0.3076,  0.8898,  0.3371,  0.3170],
         [-0.1141,  0.9374, -0.3289,  0.2450],
         [-0.9296,  0.1630, -0.3306,  0.1192],
         [ 0.9296, -0.1630,  0.3306,  0.1192],
         [-0.0419,  0.9784, -0.2024,  0.5580],
         [-0.0640,  0.9946, -0.0814,  0.4415],
         [ 0.0859,  0.9944, -0.0613,  0.5472],
         [ 0.1180,  0.9777,  0.1736,  0.4415]])

    predicted = predicted[:,:3]
    target = target[:,:3]
    plot_results(predicted, target)