import numpy as np
import cv2 as cv
from PIL import Image
from common.human import *
from common.misc import *


class Data:
    def __init__(self, npz_path, transforms=None, train=True, action=None):
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
        num_frame = 0

        if "h36m" in npz_path:
            print("INFO: Using Human3.6M dataset.")
            subject = {
                "subjects_train": ["S1/", "S5/", "S6/", "S7/", "S8/"],
                "subjects_test": ["S9/", "S11/"]
            }
            if action is None:
                if train:
                    # training
                    to_load = [item for item in data.files \
                        for S in subject["subjects_train"] if S in item]
                else:
                    # validating
                    to_load = [item for item in data.files \
                        for S in subject["subjects_test"] if S in item]
            else:
                # evaluating
                to_load = [item for item in data.files \
                    for S in subject["subjects_test"] if S in item and action in item]

            import random
            random.seed(100)
            for act in to_load:
                frames = data[act].flatten()[0]
                reduced = random.sample(list(frames), int(len(frames)*0.1)) \
                    if action is None else random.sample(list(frames), int(len(frames)))
                num_frame += len(reduced)
                for f in reduced:
                    gt_2d = self.zero_center(frames[f]["positions_2d"])
                    gt_3d = self.zero_center(self.remove_joints( \
                            frames[f]["positions_3d"], "h36m"))

                    assert gt_2d.shape == (17,2) and gt_3d.shape == (17,3)
                    self.gt_pts2d.append(gt_2d)
                    self.gt_pts3d.append(gt_3d)
                    self.gt_vecs3d.append(convert_gt(gt_3d, t_info))
                    self.img_path.append(frames[f]["directory"])

        elif "mpi_inf_3dhp_test_set" in npz_path:
            if action is not None:
                print("Only support action parameter in H3.6M dataset.")
                exit(0)
            print("INFO: Using MPI-INF-3DHP testing dataset.")
            pts_2d = data["positions_3d"].flatten()[0].keys()
            pts_3d = data["positions_2d"].flatten()[0].keys()
            img_paths = data["img_paths"].flatten()[0].keys()
            self.gt_pts2d.append()

        else:
            if action is not None:
                print("Only support action parameter in H3.6M dataset.")
                exit(0)
            print("INFO: Using MPI-INF-3DHP dataset.")
            data = data["arr_0"].reshape(1,-1)[0]
            vid_list = np.arange(6)
            if not train:
                vid_list = np.arange(6,8)

            for vid in vid_list:
                num_frame += len(data[vid].keys())
                for frame in data[vid].keys():
                    pts_2d = data[vid][frame]["pts_2d"]
                    gt_2d = self.zero_center(self.remove_joints(pts_2d))

                    pts_3d = data[vid][frame]["pts_3d"]
                    cam_3d = self.to_camera_coordinate(pts_2d, pts_3d, vid)
                    gt_3d = self.zero_center(cam_3d)/1000

                    self.gt_pts2d.append(gt_2d)
                    self.gt_pts3d.append(gt_3d)
                    self.gt_vecs3d.append((convert_gt(gt_3d, t_info)))
                    self.img_path.append(data[vid][frame]["directory"])

        print("INFO: Using ", num_frame, " frames")

    def __getitem__(self, index):
        try:
            img_path = self.img_path[index]
            #img = Image.open(img_path)
            #img = self.transforms(img)
            kpts_2d = self.gt_pts2d[index]
            kpts_3d = self.gt_pts3d[index]
            vecs_3d = self.gt_vecs3d[index]
        except:
            return None
        return img_path, kpts_2d, kpts_3d, vecs_3d
        

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
            new_skel = self.remap_h36m(new_skel)
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

    
    def zero_center(self, cam) -> np.array:
        """translate root joint to origin (0,0,0)"""
        return cam - cam[2,:]


    def remap_h36m(self, h36m_joints):
        """
        convert Human3.6M joint order
        to normal MPI-INF-3DHP one
        """
        mpi_joints = np.zeros_like(h36m_joints)
        replace_list = (
            (0,8), (1,7), (2,0), (3,9), (4,10),
            (5,11), (6,12), (7,13), (8,14), (9,15), (10,16),
            (11,4), (12,5), (13,6), (14,1), (15,2), (16,3)
        )
        for joint in replace_list:
            mpi_joints[joint[0]] = h36m_joints[joint[1]]

        return mpi_joints