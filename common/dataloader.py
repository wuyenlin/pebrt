import numpy as np
import cv2 as cv
from PIL import Image
from torchvision import transforms
try:
    from common.human import *
except ModuleNotFoundError:
    from human import *



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

        if npz_path.endswith("h36m.npz"):
            print("INFO: Training on Human3.6M dataset.")
            import random
            data_2d = data["positions_2d"].reshape(1,-1)[0][0]["S1"]
            data_3d = data["positions_3d"].reshape(1,-1)[0][0]["S1"]
            data2d_list = random.sample(data_2d.keys(), 24)
            data3d_list = random.sample(data_3d.keys(), 24)
            if not train:
                data2d_list = data_2d.keys() - data2d_list
                data3d_list = data_3d.keys() - data3d_list

            for action in data3d_list:
                for frame in range(data_3d[action].shape[0]):
                    gt_2d = self.zero_center(data_2d[action][0][frame,:,:], "h36m")
                    gt_3d = self.zero_center(self.remove_joints( \
                            data_3d[action][frame,:,:], "h36m"), "h36m")

                    self.gt_pts2d.append(gt_2d)
                    self.gt_pts3d.append(gt_3d)
                    self.gt_vecs3d.append((convert_gt(gt_3d, t_info, "h36m")))
                    self.img_path.append(frame)

            
        else:
            print("INFO: Training on MPI-INF-3DHP dataset.")
            data = data["arr_0"].reshape(1,-1)[0]
            vid_list = np.arange(6)
            if not train:
                vid_list = np.arange(6,8)

            for vid in vid_list:
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


    def __getitem__(self, index):
        try:
            frame = self.img_path[index]
            img = Image.open(img_path)
            img = self.transforms(img)
            kpts_2d = self.gt_pts2d[index]
            kpts_3d = self.gt_pts3d[index]
            vecs_3d = self.gt_vecs3d[index]
        except:
            return None
        return img_path, img, kpts_2d, vecs_3d
        

    def __len__(self):
        return len(self.gt_pts2d)
    

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


def try_load():
    from torchvision import transforms
    from torch.utils.data import DataLoader
    train_npz = "./data_h36m.npz"
    train_npz = "./dataset/S1/Seq1/imageSequence/S1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    frame, gt_2d, gt_3d, vec_3d = dataiter.next()

    
    # bones = (
    # (0,1), (0,3), (1,2), (3,4),  # spine + head
    # (0,5), (0,8),
    # (5,6), (6,7), (8,9), (9,10), # arms
    # (2,14), (2,11),
    # (11,12), (12,13), (14,15), (15,16), # legs
    # )

    # # 1st - Image
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(Image.open(img_path[0]))

    # # 2nd- 3D Pose
    # pts = vec[0]
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    # for bone in bones:
    #     xS = (pts[bone[0],0], pts[bone[1],0])
    #     yS = (pts[bone[0],1], pts[bone[1],1])
    #     zS = (pts[bone[0],2], pts[bone[1],2])
        
    #     ax.plot(xS, yS, zS)
    
    # ax.view_init(elev=-80, azim=-90)
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    # ax.set_zlim(-1,1)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")


    # plt.show()


if __name__ == "__main__":

    try_load()