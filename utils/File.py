#!/usr/bin/python3

import os
from numpy.linalg import norm
from Video import *

class File(Video):
    def __init__(self, S, Se, vid):
        super().__init__(S, Se, vid)

    def __del__(self):
        print("Killed")

    # make sure coordinates do not go beyond the frame
    def bound_number(self, x, y, frame_size):
        if x < 0 :
            x = 0
            if y < 0:
                y = 0
        elif y < 0 :
            y = 0
        elif x > frame_size.shape[0]:
            x = frame_size.shape[0]
            if y > frame_size.shape[1]:
                y = frame_size.shape[1]
        elif y > frame_size.shape[1]:
            y = frame_size.shape[1]
        return x, y

    def in_box(self, pt, start, end):
        return True if (start[0]<=pt[0]<=end[0] and start[1]<=pt[1]<=end[1]) else False

    def check_valid(self, nframe, start, end):
        """
        Verify that all 3 keypoints are within the box
        """
        self.calib(nframe)
        self.get_joints(nframe)
        root = (self.proj_xS[4], self.proj_yS[4])
        L_shoulder = (self.proj_xS[9], self.proj_yS[9])
        R_shoulder = (self.proj_xS[14], self.proj_yS[14])
        if ((self.in_box(root,start,end)) and (self.in_box(L_shoulder,start,end)) and (self.in_box(R_shoulder,start,end))):
            return True
        return False

if __name__ == "__main__":
    f = File(1,1,0)
    f.save_cropped(save_img=True, save_txt=True)