#!/usr/bin/python3

import scipy.io as sio
import cv2 as cv
import numpy as np
import os

class Joint:
    """
    Parse annot.mat and camera.calibration file
    """
    def __init__(self, S, Se, vid):
        self.S = S
        self.Se = Se
        self.vid = vid

        self.mat_path = "dataset/S{}/Seq{}/annot.mat".format(S,Se)
        self.avi_path = "dataset/S{}/Seq{}/imageSequence/video_{}.avi".format(S,Se,vid)
        # self.img_path = "dataset/S{}/Seq{}/imageSequence/video_{}".format(S,Se,vid)
        self.calib_path = "dataset/S{}/Seq{}/camera.calibration".format(S,Se)

        # self.camera = int(self.avi_path.split("/")[4].split("_")[-1]) # get camera number
        self.camera = vid # get camera number
        self.annot3D = sio.loadmat(self.mat_path)['annot3']
        self.annot2D = sio.loadmat(self.mat_path)['annot2']
        self.nframe = len(self.annot3D[self.camera][0]) # total number of frame 
        self.joint_pairs = (
            (0,1), (1,5), (1,2), (2,3), (3,4),  # spine + head
            (4,18), (4,23), (5,6), (5,8), (5,13), (6,7), 
            (8,9), (9,10),(10,11),(11,12),      # L_arm
            (13,14), (14,15), (15,16), (16,17), # R_arm
            (18,19), (19,20), (20,21), (21,22), # L_leg
            (23,24), (24,25), (25,26), (26,27) # R_leg
        )

    def __del__(self):
        print("Killed")

if __name__ == "__main__":
    j = Joint(1, 1, 0)
    print(j.annot3D[0].shape)
