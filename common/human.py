import math, cmath
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt


def general_rotation(a, b, r):
    """
    a : yaw
    b : pitch
    r : roll
    returns a rotation matrix R given yaw, pitch, and roll angles
    """
    row1 = np.array([cos(a)*cos(b), cos(a)*sin(b)*sin(r)-sin(a)*cos(r), cos(a)*sin(b)*cos(r)+sin(a)*sin(r)])
    row2 = np.array([sin(a)*cos(b), sin(a)*sin(b)*sin(r)+cos(a)*cos(r), sin(a)*sin(b)*cos(r)-cos(a)*sin(r)])
    row3 = np.array([-sin(b), cos(b)*sin(r), cos(b)*cos(r)])
    R = np.matrix([row1, row2, row3])
    assert cmath.isclose(np.linalg.det(R), 1)
    return R


class Human:
    """Implementation of Winter human model"""
    def __init__(self, H):
        self.half_face = 0.056*H
        self.neck = 0.062*H
        self.upper_spine, self.lower_spine = 0.144*H, 0.144*H
        self.clavicle = 0.129*H
        self.upper_arm, self.lower_arm = 0.188*H, 0.145*H

        self.pelvis = 0.095*H
        self.thigh, self.calf = 0.245*H, 0.246*H

        self.root = np.array([0.0, 0.0, 0.0])
        self.bones = (
        (0,1), (0,3), (1,2), (3,4),  # spine + head
        (0,5), (0,8),
        (5,6), (6,7), (8,9), (9,10), # arms
        (2,14), (2,11),
        (11,12), (12,13), (14,15), (15,16), # legs
        )
    
    def init_pose(self):
        root = self.root
        mid_spine = np.array([0, self.lower_spine, 0])
        neck = np.array([0, self.upper_spine, 0]) + mid_spine
        chin = np.array([0, self.neck, 0]) + neck
        nose = np.array([0, self.half_face, 0]) + chin

        l_shoulder = np.array([self.clavicle, 0, 0]) + neck
        l_elbow = np.array([self.upper_arm, 0, 0]) + l_shoulder
        l_wrist = np.array([self.lower_arm, 0, 0]) + l_elbow
        r_shoulder = np.array([-self.clavicle, 0, 0]) + neck
        r_elbow = np.array([-self.upper_arm, 0, 0]) + r_shoulder
        r_wrist = np.array([-self.lower_arm, 0, 0]) + r_elbow

        l_hip = np.array([self.pelvis/2, 0, 0])
        l_knee = np.array([0, -self.thigh, 0]) + l_hip
        l_ankle = np.array([0, -self.calf, 0]) + l_knee
        r_hip = np.array([-self.pelvis/2, 0, 0])
        r_knee = np.array([0, -self.thigh, 0]) + r_hip
        r_ankle = np.array([0, -self.calf, 0]) + r_knee
        model = [neck, mid_spine, root, chin, nose,
                l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist,
                l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
        self.model = model
    
    def vis_model(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p in self.model:
            ax.scatter(p[0], p[1], p[2], c='r')

        # output = self.model
        # for bone in self.bones:
        #     xS = (output[:,bone[0],0].tolist()[0], output[:,bone[1],0].tolist()[0])
        #     yS = (output[:,bone[0],1].tolist()[0], output[:,bone[1],1].tolist()[0])
        #     zS = (output[:,bone[0],2].tolist()[0], output[:,bone[1],2].tolist()[0])
            # ax.plot(xS, yS, zS)
        # ax.view_init(elev=-80, azim=-90)
        ax.view_init(elev=100, azim=-90)
        ax.autoscale()
        # plt.gca().invert_yaxis()
        plt.xlabel("X axis label")
        plt.ylabel("Y axis label")
        plt.show()
        
    def _get_bones(self):
        """get bones as vectors"""
        pass

    def fk(self):
        pass

if __name__ == "__main__":
    h = Human(1.8)
    h.init_pose()
    h.vis_model()

