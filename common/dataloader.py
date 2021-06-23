import numpy as np
import cv2 as cv
from PIL import Image
from torchvision import transforms
from common.human import *


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_rot_from_vecs(vec1: np.array, vec2: np.array) -> np.array:
    """ 
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector

    :return R: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    
    Such that vec2 = R @ vec1
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R


def convert_gt(gt_3d: np.array, t_info, dataset="mpi") -> np.array:
    """
    Compare GT3D kpts with T pose and obtain 16 rotation matrices

    :return R_stack: a (16,9) arrays with flattened rotation matrix for 16 bones
    """
    # process GT
    bone_info = vectorize(gt_3d, dataset)[:,:3] # (16,3) bone vecs

    num_row = bone_info.shape[0]
    R_stack = np.zeros([num_row, 9])
    # get rotation matrix for each bone
    for k in range(num_row):
        R = get_rot_from_vecs(t_info[k,:], bone_info[k,:]).flatten()
        R_stack[k,:] = R
    return R_stack


class Data:
    def __init__(self, npz_path, transforms=None, train=True):
        self.img_path = []
        self.gt_pts2d = []
        self.gt_pts3d = []
        self.gt_vecs3d = []
        self.transforms = transforms

        # T pose
        h = Human(1.8, "cpu")
        model = h.update_pose()
        t_info = vectorize(model)[:,:3]

        data = np.load(npz_path, allow_pickle=True)

        if "h36m" in npz_path:
            print("INFO: Using Human3.6M dataset.")
            import random
            S = [item for item in data.files if "S1/" in item]
            # S = [item for item in data.files]
            train_data = random.sample(S, int(len(S)*0.7)) if train \
                else random.sample(S, int(len(S)*0.3))

            for action in train_data:
                frames = data[action].flatten()[0]
                for f in frames:
                    gt_2d = self.zero_center(frames[f]["positions_2d"], "h36m")
                    gt_3d = self.zero_center(self.remove_joints( \
                            frames[f]["positions_3d"], "h36m"), "h36m")

                    assert gt_2d.shape == (17,2) and gt_3d.shape == (17,3)
                    self.gt_pts2d.append(gt_2d)
                    self.gt_pts3d.append(gt_3d)
                    self.img_path.append(frames[f]["directory"])

        else:
            print("INFO: Using MPI-INF-3DHP dataset.")
            data = data["arr_0"].reshape(1,-1)[0]
            vid_list = np.arange(6)
            if not train:
                vid_list = np.arange(6,8)

            for vid in vid_list:
                for frame in data[vid].keys():
                    pts_2d = data[vid][frame]["positions_2d"]
                    gt_2d = self.zero_center(self.remove_joints(pts_2d))

                    pts_3d = data[vid][frame]["positions_3d"]
                    cam_3d = self.to_camera_coordinate(pts_2d, pts_3d, vid)
                    gt_3d = self.zero_center(cam_3d)/1000

                    self.gt_pts2d.append(gt_2d)
                    self.gt_pts3d.append(gt_3d)
                    self.gt_vecs3d.append((convert_gt(gt_3d, t_info)))
                    self.img_path.append(data[vid][frame]["directory"])


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
    

    def remove_joints(self, kpts, dataset="mpi"):
        """
        Get 17 joints from the original 28 (MPI) / 32 (Human3.6M)
        """
        new_skel = np.zeros([17,3]) if kpts.shape[-1]==3 else np.zeros([17,2])
        if dataset == "mpi":
            keep = [2,4,5,6,         # spine+head
                    9,10,11,14,15,16,  # arms
                    18,19,20,23,24,25] # legs
            for row in range(17):
                new_skel[row, :] = kpts[keep[row-1], :]
            # interpolate clavicles to obtain vertebra
            new_skel[0, :] = (new_skel[5,:]+new_skel[8,:])/2
        elif dataset == "h36m":
            keep = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
            for row in range(17):
                new_skel[row, :] = kpts[keep[row], :]
        else:
            print("Unrecognized dataset name.")
        return new_skel


    def get_intrinsic(self, camera):
        """
        Parse camera matrix from calibration file
        :param camera: camera number (used in MPI dataset)
        :return intrinsic matrix:
        """
        calib = open("./dataset/S1/Seq1/camera.calibration","r")
        content = calib.readlines()
        content = [line.strip() for line in content]
        # 3x3 intrinsic matrix
        intrinsic = np.array(content[7*camera+5].split(" ")[3:], dtype=np.float32)
        self.intrinsic = intrinsic.reshape(4,-1)[:3, :3]


    def to_camera_coordinate(self, pts_2d, pts_3d, camera) -> np.array:
        self.get_intrinsic(camera)
        ret, R, t = cv.solvePnP(pts_3d, pts_2d, self.intrinsic, np.zeros(4), flags=cv.SOLVEPNP_EPNP)

        # get extrinsic matrix
        assert ret
        R = cv.Rodrigues(R)[0]
        E = np.concatenate((R,t), axis=1)  # [R|t], a 3x4 matrix
    
        pts_3d = cv.convertPointsToHomogeneous(self.remove_joints(pts_3d)).transpose().squeeze(1)
        cam_coor = E @ pts_3d
        cam_3d = cam_coor.transpose()
        return cam_3d

    
    def zero_center(self, cam, dataset="mpi") -> np.array:
        """translate root joint to origin (0,0,0)"""
        if dataset == "mpi":
            return cam - cam[2,:]
        elif dataset == "h36m":
            return cam - cam[0,:]
        else:
            print("Unrecognized dataset name.")
