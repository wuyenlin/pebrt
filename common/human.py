import math, cmath
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt

try:
    from common.misc import *
except ModuleNotFoundError:
    from misc import *


class Human:
    """Implementation of Winter human model"""
    def __init__(self, H):
        self.half_face = 0.066*H
        self.neck = 0.052*H
        self.upper_spine, self.lower_spine = 0.144*H, 0.144*H
        self.clavicle = 0.129*H
        self.upper_arm, self.lower_arm = 0.186*H, 0.146*H

        self.pelvis = 0.191*H
        self.thigh, self.calf = 0.245*H, 0.246*H

        self.root = np.array([0.0, 0.0, 0.0])
        self.indices = (
        (0,1), (0,3), (1,2), (3,4),  # spine + head
        (0,5), (0,8),
        (5,6), (6,7), (8,9), (9,10), # arms
        (2,14), (2,11),
        (11,12), (12,13), (14,15), (15,16), # legs
        )
    

    def _init_bones(self):
        """get bones as vectors"""
        self.bones = {
            'lower_spine': np.array([0, -self.lower_spine, 0]),
            'upper_spine': np.array([0, -self.upper_spine, 0]),
            'neck': np.array([0, -self.neck, 0]),
            'head': np.array([0, -self.half_face, 0]),

            'l_clavicle': np.array([self.clavicle, 0, 0]),
            'l_upper_arm': np.array([self.upper_arm, 0, 0]),
            'l_lower_arm': np.array([self.lower_arm, 0, 0]),
            'r_clavicle': np.array([-self.clavicle, 0, 0]),
            'r_upper_arm': np.array([-self.upper_arm, 0, 0]),
            'r_lower_arm': np.array([-self.lower_arm, 0, 0]),

            'l_hip': np.array([self.pelvis/2, 0, 0]),
            'l_thigh': np.array([0, self.thigh, 0]),
            'l_calf': np.array([0, self.calf, 0]),
            'r_hip': np.array([-self.pelvis/2, 0, 0]),
            'r_thigh': np.array([0, self.thigh, 0]),
            'r_calf': np.array([0, self.calf, 0])
        }


    def _sort_angles(self, ang):
        """process the PETR output (26 angles in rad) and sort them in a dict"""
        self.angles = {
            'lower_spine': rot(ang[0], ang[1], ang[2]),
            'upper_spine': rot(ang[3], ang[4], ang[5]),
            'neck': rot(ang[6], ang[7], ang[8]),
            'head': rot(ang[9], ang[10], ang[11]),

            'l_upper_arm': rot(ang[12], ang[13], ang[14]),
            'l_lower_arm': rot(ang[15], ang[16], 0),
            'r_upper_arm': rot(ang[17], ang[18], ang[19]),
            'r_lower_arm': rot(ang[20], ang[21], 0),

            'l_thigh': rot(ang[22], ang[23], ang[24]),
            'l_calf': rot(ang[25], 0, 0),
            'r_thigh': rot(ang[26], ang[27], ang[28]),
            'r_calf': rot(ang[29], 0, 0)
        }


    def update_bones(self, ang=None):
        self._init_bones()
        if ang is not None:
            self._sort_angles(ang)
            for bone in self.angles.keys():
                self.bones[bone] = self.angles[bone] @ self.bones[bone]


    def update_pose(self, ang=None):
        """
        returns a numpy array of (17,3)
        """
        self.update_bones(ang)

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
        model = np.array([neck, lower_spine, root, chin, nose,
                l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist,
                l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle])
        
        self.model = model
        return model


    def vis_model(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p in self.model:
            ax.scatter(p[0], p[1], p[2], c='r')

        for index in self.indices:
            xS = (self.model[index[0]][0], self.model[index[1]][0])
            yS = (self.model[index[0]][1], self.model[index[1]][1])
            zS = (self.model[index[0]][2], self.model[index[1]][2])
            ax.plot(xS, yS, zS)
        ax.view_init(elev=-80, azim=-90)
        ax.autoscale()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


if __name__ == "__main__":
    h = Human(1.8)

    a = np.random.rand(30)
    model = h.update_pose(a)
    print(model.shape)
    h.vis_model()