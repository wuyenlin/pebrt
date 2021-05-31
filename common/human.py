import numpy as np
import cmath
import torch


def rot(euler) -> torch.tensor:
    """
    General rotation matrix
    :param a: yaw (rad) - rotation along z axis
    :param b: pitch (rad) - rotation along y axis
    :param r: roll (rad) - rotation along x axis
    
    :return R: a rotation matrix R
    """
    from math import sin, cos
    a, b, r = euler[0], euler[1], euler[2]
    row1 = torch.tensor([cos(a)*cos(b), cos(a)*sin(b)*sin(r)-sin(a)*cos(r), cos(a)*sin(b)*cos(r)+sin(a)*sin(r)])
    row2 = torch.tensor([sin(a)*cos(b), sin(a)*sin(b)*sin(r)+cos(a)*cos(r), sin(a)*sin(b)*cos(r)-cos(a)*sin(r)])
    row3 = torch.tensor([-sin(b), cos(b)*sin(r), cos(b)*cos(r)])
    R = torch.stack((row1, row2, row3), 0)
    assert cmath.isclose(torch.det(R), 1, rel_tol=1e-04), torch.det(R)
    return R.flatten()


def euler_from_rot(R: np.array) -> np.array:
    import cv2 as cv
    angles = cv.RQDecomp3x3(R)[0]
    return np.radians(angles)


class Human:
    """
    Implementation of Winter human model
    """
    def __init__(self, H, device="cuda:0"):
        self.device = device
        self.half_face = 0.066*H
        self.neck = 0.052*H
        self.upper_spine, self.lower_spine = 0.144*H, 0.144*H
        self.clavicle = 0.129*H
        self.upper_arm, self.lower_arm = 0.186*H, 0.146*H

        self.pelvis = 0.191*H
        self.thigh, self.calf = 0.245*H, 0.246*H
        self.root = torch.zeros(3, device=self.device)

        self.constraints = {
            'lower_spine': ((-0.52,1.31), (-0.52,0.52), (-0.61,0.61)),
            'upper_spine': ((-0.52,1.57), (-0.52,0.52), (-0.61,0.61)),
            'neck': ((-0.872,1.39), (-1.22,1.22), (-0.61,0.61)),
            'head': ((-0.872,1.39), (-1.22,1.22), (-0.61,0.61)),

            'l_clavicle': ((0,0), (0,0), (0,0)),
            'l_upper_arm': ((0,0), (-0.707,2.27), (-1.57,2.28)),
            'l_lower_arm': ((0,0), (-0.707,2.27), (-4.19,2.28)),
            'r_clavicle': ((0,0), (0,0), (0,0)),
            'r_upper_arm': ((0,0), (-2.27,0.707), (-2.28,1.57)),
            'r_lower_arm': ((0,0), (-2.27,0.707), (-2.28,4.19)),

            'l_hip': ((0,0), (0,0), (0,0)),
            'l_thigh': ((-2.09,0.52), (0,0), (-0.87,0.35)),
            'l_calf': ((-2.09,2.79), (0,0), (-0.87,0.35)),
            'r_hip': ((0,0), (0,0), (0,0)),
            'r_thigh': ((-0.52,2.09), (0,0), (-0.35,0.87)),
            'r_calf': ((-2.09,2.79), (0,0), (-0.35,0.87)),
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
        self.bones = {bone: self.bones[bone].to(self.device) for bone in self.bones.keys()}
        

    def check_constraints(self, bone, R: np.array):
        """
        This function punishes if NN outputs are beyond joint rotation constraints.
        """
        punish_w = 1
        euler_angles = euler_from_rot(np.array(R).reshape(3,-1))
        for i in range(3):
            low = self.constraints[bone][i][0]
            high = self.constraints[bone][i][1]
            if euler_angles[i] < low:
                euler_angles[i] = low
                punish_w += 0.5
            elif euler_angles[i] > high:
                euler_angles[i] = high
                punish_w += 0.5
        # sort angles in z-y-x order for rot function
        euler_angles[0], euler_angles[2] = euler_angles[2], euler_angles[0]
        return rot(euler_angles), punish_w


    def sort_rot(self, elem):
        """
        :param ang: a list of 144 elements (9 * 16)
        process PETRA output to rotation matrix of 16 bones
        """
        import torch.nn.functional as f
        elem = elem.flatten()
        self.rot_mat, self.punish_list = {}, []
        k = 0
        for bone in self.constraints.keys():
            R = elem[9*k:9*(k+1)]
            # R, punish_w = self.check_constraints(bone, R)
            # self.punish_list.append(punish_w)

            # R = f.normalize(R.to(torch.float32).view(3,-1))
            R = f.normalize(torch.tensor(R, dtype=torch.float32).view(3,-1))
            assert cmath.isclose(torch.linalg.det(R), 1, rel_tol=1e-04), torch.linalg.det(R)
            self.rot_mat[bone] = R
            k += 1


    def update_bones(self, elem=None):
        """
        Initiates a T-Pose human model and rotate each bone using the given rotation matrices
        :return model: a numpy array of (17,3)
        """
        self._init_bones()
        if elem is not None:
            self.sort_rot(elem)
            self.bones = {bone: self.rot_mat[bone] @ self.bones[bone] for bone in self.constraints.keys()}


    def update_pose(self, elem=None, debug=False):
        """
        Assemble bones to make a human body
        """
        self.update_bones(elem)
        if debug:
            for bone in self.constraints.keys():
                print(bone, ":\n", self.rot_mat[bone])

        root = self.root
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


def vectorize(gt_3d) -> torch.tensor:
    """
    process gt_3d (17,3) into a (16,4) that contains bone vector and length
    :return bone_info: [unit bone vector (,3) + bone length (,1)]
    """
    indices = (
        (2,1), (1,0), (0,3), (3,4),  # spine + head
        (0,5), (5,6), (6,7), 
        (0,8), (8,9), (9,10), # arms
        (2,14), (11,12), (12,13),
        (2,11), (14,15), (15,16), # legs
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
        unit_vec = vec/vec_len
        bone_info[i,:3], bone_info[i,3] = unit_vec, vec_len
    return bone_info


def vis_model(model):
    indices = (
        (2,1), (1,0), (0,3), (3,4),  # spine + head
        (0,5), (5,6), (6,7), 
        (0,8), (8,9), (9,10), # arms
        (2,14), (11,12), (12,13),
        (2,11), (14,15), (15,16), # legs
    )
    import matplotlib.pyplot as plt
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
    ax.set_zlim([-1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def rand_pose():
    h = Human(1.8, "cpu")
    euler = (0,0,0)
    a = rot(euler).repeat(16)
    model = h.update_pose(a)
    print(model)
    print(h.punish_list)
    vis_model(model)


if __name__ == "__main__":
    rand_pose()