import math, cmath
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2 as cv


def normalize(x: torch.tensor) -> torch.tensor:
    return x/torch.linalg.norm(x)


def rotation_matrix_from_vectors(vec1: np.array, vec2: np.array) -> np.array:
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector

    :return R: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    
    Such that b = R @ a

    (Credit to Peter from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space)
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R


def get_euler(R: np.array) -> tuple:
    return cv.RQDecomp3x3(R)[0]


class Human:
    """
    Implementation of Winter human model
    """
    def __init__(self, H, device="cuda:0"):
        self.half_face = 0.066*H
        self.neck = 0.052*H
        self.upper_spine, self.lower_spine = 0.144*H, 0.144*H
        self.clavicle = 0.129*H
        self.upper_arm, self.lower_arm = 0.186*H, 0.146*H

        self.pelvis = 0.191*H
        self.thigh, self.calf = 0.245*H, 0.246*H
        self.root = torch.zeros(3)
        self.device = device

        self.constraints = {
            'lower_spine': ((-1.0,1.0), (0,0), (-1.0,1.0)),
            'upper_spine': ((-2.0,2.0), (0,0), (-2.0,2.0)),

            'neck': ((-1.0,1.0), (0,0), (0,0)),
            'head': ((-1.0,1.0), (0,0), (-1.0,1.57)),

            'l_upper_arm': ((-3.14,3.14), (-0.1,1.7), (0,0)),
            'l_lower_arm': ((-6.14,3.14), (0,0), (0,0)),
            'r_upper_arm': ((-3.14,3.14), (-1.7,0.1), (0,0)),
            'r_lower_arm': ((-3.14,6.14), (0,0), (0,0)),

            'l_thigh': ((-1.57,1.0), (0,0), (-1.57,1.57)),
            'l_calf': ((-1.57,1.0), (0,0), (-4.71,1.57)),
            'r_thigh': ((-1.0,1.57), (0,0), (-1.57,1.57)),
            'r_calf': ((-1.0,1.57), (0,0), (-4.71,1.57)),
        }


    def _init_bones(self):
        """get bones as vectors"""
        self.bones = {
            'lower_spine': torch.tensor([0, -self.lower_spine, 0]),
            'upper_spine': torch.tensor([0, -self.upper_spine, 0]),
            'neck': torch.tensor([0, -self.neck, 0]),
            'head': torch.tensor([0, -self.half_face, 0]),

            'l_clavicle': torch.tensor([self.clavicle, 0, 0]),
            'l_upper_arm': torch.tensor([self.upper_arm, 0, 0]),
            'l_lower_arm': torch.tensor([self.lower_arm, 0, 0]),
            'r_clavicle': torch.tensor([-self.clavicle, 0, 0]),
            'r_upper_arm': torch.tensor([-self.upper_arm, 0, 0]),
            'r_lower_arm': torch.tensor([-self.lower_arm, 0, 0]),

            'l_hip': torch.tensor([self.pelvis/2, 0, 0]),
            'l_thigh': torch.tensor([0, self.thigh, 0]),
            'l_calf': torch.tensor([0, self.calf, 0]),
            'r_hip': torch.tensor([-self.pelvis/2, 0, 0]),
            'r_thigh': torch.tensor([0, self.thigh, 0]),
            'r_calf': torch.tensor([0, self.calf, 0])
        }
        for bone in self.bones.keys():
            self.bones[bone] = self.bones[bone].to(self.device)
        

    def convert_gt_6d(self, gt_3d: np.array) -> torch.tensor:
        """
        Compare GT3D kpts with T pose and obtain the 6D array for SO(3)
        """
        self._init_bones()
        self.rot_mat = {k: rotation_matrix_from_vectors(v) for k,v in self.bones.items()}
        #     self.check_constraints()
        


    def check_constraints(self):
        """
        This function punishes if NN outputs are beyond joint rotation constraints.
        """
        punish = {}

        for bone in self.constraints.keys():
            count = 1
            for i in range(3):
                low = self.constraints[bone][i][0]
                high = self.constraints[bone][i][1]
                if self.angles[bone][i] < low or self.angles[bone][i] > high:
                    count += 0.5
            punish[bone] = count
        
        self.punish_list = [punish[list(punish.keys())[k]] for k in range(len(punish.keys()))]

        self.punish_list.insert(4, 1)
        self.punish_list.insert(4, 1)
        self.punish_list.insert(10, 1)
        self.punish_list.insert(10, 1)


    def gschmidt(self, arr: torch.tensor) -> torch.tensor:
        """
        an implementation of 6D representation for 3D rotation using Gram-Schmidt process

        :return R: 3x3 rotation matrix
        """
        arr = arr.to(torch.float32)
        a_1, a_2 = arr[:3], arr[3:]
        row_1 = normalize(a_1)
        row_2 = normalize(a_2 - (row_1@a_2)*row_1)
        row_3 = normalize(torch.cross(row_1,row_2))
        R = torch.stack((row_1, row_2, row_3), 1) # SO(3)
        assert cmath.isclose(torch.det(R), 1, rel_tol=1e-04), torch.det(R)
        return R.to(self.device)


    def sort_angles(self, ang):
        """
        :param ang: a list of 48 elements
        process PETRA output (quaternion ele) to rotation matrix of 12 bones
        """
        self.angles = {}
        k = 0
        for bone in self.constraints.keys():
            self.angles[bone] = ang[6*k:6*(k+1)]
            k += 1


    def update_bones(self, ang=None):
        """
        :return model: a numpy array of (17,3)
        """
        self._init_bones()
        if ang is not None:
            self.sort_angles(ang)
            self.rot_mat = {k: self.gschmidt(v) for k,v in self.angles.items()}
            self.check_constraints()
            for bone in self.angles.keys():
                self.bones[bone] = self.rot_mat[bone] @ self.bones[bone]


    def update_pose(self, ang=None):
        """
        Assemble bones to make a human body
        """
        self.update_bones(ang)

        root = self.root.to(self.device)
        lower_spine = self.bones['lower_spine']
        neck = self.bones['upper_spine'] + lower_spine
        chin = self.bones['neck'] + neck
        nose = self.bones['head'] + chin

        l_shoulder = self.bones['l_clavicle'] + neck
        l_elbow = self.bones['l_upper_arm'] + l_shoulder
        l_wrist = self.bones['l_lower_arm'] + l_elbow
        r_shoulder = self.bones['r_clavicle'] + neck
        r_elbow = self.bones['r_upper_arm'] + r_shoulder
        r_wrist = self.bones['r_lower_arm'] + r_elbow

        l_hip = self.bones['l_hip']
        l_knee = self.bones['l_thigh'] + l_hip
        l_ankle = self.bones['l_calf'] + l_knee
        r_hip = self.bones['r_hip']
        r_knee = self.bones['r_thigh'] + r_hip
        r_ankle = self.bones['r_calf'] + r_knee
        self.model = torch.stack((neck, lower_spine, root, chin, nose,
                l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist,
                l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle), 0)
        
        return self.model


def vis_model(model):
    indices = (
        (0,1), (0,3), (1,2), (3,4),  # spine + head
        (0,5), (0,8), # clavicle
        (5,6), (6,7), (8,9), (9,10), # arms
        (2,14), (2,11), # pelvis
        (11,12), (12,13), (14,15), (15,16), # legs
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in model:
        ax.scatter(p[0], p[1], p[2], c='r')

    for index in indices:
        xS = (model[index[0]][0], model[index[1]][0])
        yS = (model[index[0]][1], model[index[1]][1])
        zS = (model[index[0]][2], model[index[1]][2])
        ax.plot(xS, yS, zS)
    ax.view_init(elev=-80, azim=-90)
    ax.autoscale()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def vectorize(gt_3d) -> torch.tensor:
    """
    process gt_3d (17,3) into a (16,4) that contains bone vector and length
    :return bone_info: unit bone vector + bone length 
    """
    indices = (
        (0,1), (0,3), (1,2), (3,4),  # spine + head
        (0,5), (0,8), # clavicle
        (5,6), (6,7), (8,9), (9,10), # arms
        (2,14), (2,11), # pelvis
        (11,12), (12,13), (14,15), (15,16), # legs
    )
    num_bones = len(indices)
    try:
        gt_3d_tensor = torch.from_numpy(gt_3d)
    except TypeError:
        gt_3d_tensor = gt_3d
    bone_info = torch.zeros([num_bones, 4], requires_grad=False) # (16, 4)
    for i in range(num_bones):
        vec = gt_3d_tensor[indices[i][1],:] - gt_3d_tensor[indices[i][0],:]
        vec_len = torch.linalg.norm(vec)
        unit_vec = normalize(vec)
        bone_info[i,:3], bone_info[i,3] = unit_vec, vec_len
    return bone_info


def rand_pose():
    h = Human(1.8, "cpu")
    a = torch.rand(72)
    model = h.update_pose(a)
    print(model)
    print(h.punish_list)
    vis_model(model)
    print(vectorize(model))


if __name__ == "__main__":
    rand_pose()
