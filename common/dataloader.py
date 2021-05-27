from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Data:
    def __init__(self, npz_path, transforms=None, train=True):
        data = np.load(npz_path, allow_pickle=True)
        data = data["arr_0"].reshape(1,-1)[0]

        self.img_path = []
        self.gt_pts2d = []
        self.gt_pts3d = []
        self.gt_vecs3d = []
        self.transforms = transforms

        if train:
            vid_list = np.arange(6)
        else:
            vid_list = np.arange(6,8)

        for vid in vid_list:
            for frame in data[vid].keys():
                bbox_start = data[vid][frame]["bbox_start"]
                pts_2d = (data[vid][frame]["pts_2d"])
                gt_2d = self.pop_joints(pts_2d) - bbox_start

                pts_3d = (data[vid][frame]["pts_3d"])
                cam_3d = (data[vid][frame]["cam_3d"])
                gt_3d = self.zero_center(self.pop_joints(cam_3d))/1000

                vec_3d = (data[vid][frame]["vec_3d"])

                self.gt_pts2d.append(gt_2d)
                self.gt_pts3d.append(gt_3d)
                self.gt_vecs3d.append(vec_3d)
                self.img_path.append(data[vid][frame]['directory'])

    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            kpts_3d = self.gt_pts3d[index]
            vecs_3d = self.gt_vecs3d[index]
        except:
            return None
        return img_path, img, kpts_3d, vecs_3d

    def __len__(self):
        return len(self.img_path)
    

    def pop_joints(self, kpts):
        """
        Get 17 joints from the original 28 
        :param kpts: orginal kpts from MPI-INF-3DHP (an array of (28,3))
        :return new_skel: 
        """
        new_skel = np.zeros([17,3]) if kpts.shape[-1]==3 else np.zeros([17,2])
        ext_list = [2,4,5,6,         # spine+head
                    9,10,11,14,15,16,  # arms
                    18,19,20,23,24,25] # legs
        for row in range(1,17):
            new_skel[row, :] = kpts[ext_list[row-1], :]
        # interpolate clavicles to obtain vertebra
        new_skel[0, :] = (new_skel[5,:]+new_skel[8,:])/2
        return new_skel


    def zero_center(self, cam) -> np.array:
        """
        translate root joint to origin (0,0,0)
        """
        return cam - cam[2,:]


def try_load():
    train_npz = "dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, gt3d, vec = dataiter.next()

    
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    # 1st - Image
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(Image.open(img_path[0]))

    # 2nd- 3D Pose
    pts = gt3d[0]
    ax = fig.add_subplot(1, 2, 2, projection='3d')
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


    plt.show()


if __name__ == "__main__":

    transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    try_load()
