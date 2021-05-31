import torch
import numpy as np
from PIL import Image
try:
    from common.human import *
except ModuleNotFoundError:
    from human import *


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Data:
    def __init__(self, npz_path, transforms=None, train=True):
        data = np.load(npz_path, allow_pickle=True)
        data = data["arr_0"].reshape(1,-1)[0]

        self.img_path = []
        self.gt_pts2d = []
        self.gt_vecs3d = []
        self.transforms = transforms

        if train:
            vid_list = np.arange(6)
        else:
            vid_list = np.arange(6,8)


        h = Human(1.8, "cpu")
        model = h.update_pose()
        t_info = vectorize(model)[:,:3]
        for vid in vid_list:
            for frame in data[vid].keys():
                bbox_start = data[vid][frame]['bbox_start']
                pts_2d = (data[vid][frame]["pts_2d"])
                # original
                gt_2d = self.zero_center(self.pop_joints(pts_2d))/2048
                # gt_2d = self.pop_joints(pts_2d)                    #[X]
                # gt_2d = self.pop_joints(pts_2d) - bbox_start       #[O]
                # gt_2d = self.zero_center(self.pop_joints(pts_2d))  #[O]
                # gt_2d = self.vec2d(self.pop_joints(pts_2d))

                pts_3d = (data[vid][frame]["pts_3d"])
                cam_3d = (data[vid][frame]["cam_3d"])
                gt_3d = self.zero_center(cam_3d)/1000

                vec_3d = (data[vid][frame]["vec_3d"])

                self.gt_pts2d.append(gt_2d)
                self.gt_vecs3d.append(vec_3d)
                self.img_path.append(data[vid][frame]['directory'])


    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            kpts_2d = self.gt_pts2d[index]
            vecs_3d = self.gt_vecs3d[index]
        except:
            return None
        return img_path, img, kpts_2d, vecs_3d

    def __len__(self):
        return len(self.img_path)
    

    def pop_joints(self, kpts):
        """
        Get 17 joints from the original 28 
        :param kpts: orginal kpts from MPI-INF-3DHP (an array of (28,n))
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


    def vec2d(self, input_2d):
        indices = (
            (2,1), (1,0), (0,3), (3,4),  # spine + head
            (0,5), (5,6), (6,7), 
            (0,8), (8,9), (9,10), # arms
            (2,14), (11,12), (12,13),
            (2,11), (14,15), (15,16), # legs
        )
        num_bones = len(indices)

        bone_info = np.zeros([num_bones, 2]) # (16, 2)
        for i in range(num_bones):
            vec = input_2d[indices[i][1],:] - input_2d[indices[i][0],:]
            bone_info[i,:] = vec
        return bone_info

if __name__ == "__main__":
    from torchvision import transforms
    transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    train_npz = "dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, kpts, labels = dataiter.next()
        