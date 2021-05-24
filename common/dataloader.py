from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
try:
    from common.human import *
    from common.peltra import *
except ModuleNotFoundError:
    from human import *
    from peltra import *


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

    (Credit to Peter from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space)
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R


def convert_gt(gt_3d: np.array, t_info) -> np.array:
    """
    Compare GT3D kpts with T pose and obtain 16 rotation matrices

    :return R_stack: a (16,9) arrays with flattened rotation matrix for 16 bones
    """
    # process GT
    bone_info = vectorize(gt_3d)[:,:3] # (16,3) bone vecs

    num_row = bone_info.shape[0]
    R_stack = np.zeros([num_row, 9])
    # get rotation matrix for each bone
    for k in range(num_row):
        R = get_rot_from_vecs(t_info[k,:], bone_info[k,:]).flatten()
        R_stack[k,:] = R
        
    return R_stack


class Data:
    def __init__(self, npz_path, transforms = None, train=True):
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

        # T pose
        h = Human(1.8, "cpu")
        model = h.update_pose()
        t_info = vectorize(model)[:,:3]

        for vid in vid_list:
            for frame in data[vid].keys():
                pts_2d = (data[vid][frame]['2d_keypoints']).reshape(-1,2)
                gt_2d = self.zero_center(self.pop_joints(pts_2d))/2048

                pts_3d = (data[vid][frame]['3d_keypoints']).reshape(-1,3)
                cam_3d = self.to_camera_coordinate(pts_2d, pts_3d, vid)
                gt_3d = self.zero_center(cam_3d)/1000

                self.gt_pts2d.append(gt_2d)
                self.gt_vecs3d.append((convert_gt(gt_3d, t_info)))
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
        :param kpts: orginal kpts from MPI-INF-3DHP (an array of (28,3))
        :return new_skel: 
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
        :param camera: camera number (used in MPI dataset)
        :return intrinsic matrix:
        """
        calib = open("./dataset/S1/Seq1/camera.calibration","r")
        content = calib.readlines()
        content = [line.strip() for line in content]
        # 3x3 intrinsic matrix
        intrinsic = np.array(content[7*camera+5].split(" ")[3:], dtype=np.float32)
        intrinsic = np.reshape(intrinsic, (4,-1))
        self.intrinsic = intrinsic[:3, :3]


    def to_camera_coordinate(self, pts_2d, pts_3d, camera) -> np.array:
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

    
    def zero_center(self, cam) -> np.array:
        """
        translate root joint to origin (0,0,0)
        """
        return cam - cam[2,:]


def try_load(model=False):
    train_npz = "dataset/S1/Seq1/imageSequence/S1Seq1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
#    img_path, images, kpts, labels = dataiter.next()
    img_path, kpt_2d, kpts, labels = dataiter.next()

    
    row = 3 if model else 2
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    # 1st - Image
    fig = plt.figure()
    ax = fig.add_subplot(1, row, 1)
    plt.imshow(Image.open(img_path[0]))

    # 2nd- 3D Pose
    pts = kpts[0]
    ax = fig.add_subplot(1, row, 2, projection='3d')
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

    if model:
        h = Human(1.8, "cpu")
        net = PELTRA("cuda:0")
        net.load_state_dict(torch.load('./angle_checkpoint/epoch_40.bin')['model'])
        net = net.cuda()
        net.eval()

        pts = kpt_2d[0]
        pts = torch.tensor(pts)
        pts = torch.tensor(pts.unsqueeze(0)).cuda()
        output = net(pts)

        output = h.update_pose(output)
        output = output.detach().numpy()

        ax = fig.add_subplot(1, row, 3, projection='3d')
        for p in output:
            ax.scatter(p[0], p[1], p[2], c='r')

        for index in bones:
            xS = (output[index[0]][0],output[index[1]][0])
            yS = (output[index[0]][1],output[index[1]][1])
            zS = (output[index[0]][2],output[index[1]][2])
            ax.plot(xS, yS, zS)
        ax.view_init(elev=-80, azim=-90)
        ax.autoscale()
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.show()
    plt.savefig('./angle_checkpoint/this.svg', format='svg', dpi=1200)


if __name__ == "__main__":

    transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    try_load(model=True)
