#!/usr/bin/python3

import os, json, csv, sys
from numpy.linalg import norm
import numpy as np 
from tqdm import tqdm
from Video import *

class All(Video):
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

    def save_cropped(self, save_img=False, save_txt=True, full=False):
        data = {}
        cap = cv.VideoCapture(self.avi_path)
        if (cap.isOpened()==False):
            print("Error opening the video file.")

        try:
            if full:
                os.mkdir("dataset/S{}/Seq{}/imageSequence/full_video_{}".format(self.S, self.Se, self.vid))
            else:
                os.mkdir("dataset/S{}/Seq{}/imageSequence/video_{}".format(self.S, self.Se, self.vid))
        except FileExistsError:
            pass

        while (cap.isOpened()):
            print("processing S{} Seq{} video_{}".format(self.S, self.Se, self.vid))
            for k in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
                ret, frame = cap.read()
                if ret:
                    x1, y1, x2, y2 = self.draw_bbox(k)
                    x1, y1 = self.bound_number(x1, y1, frame)
                    x2, y2 = self.bound_number(x2, y2, frame)
                    start = (x1, y1)
                    end = (x2, y2)
                    if not self.check_valid(k, start, end): 
                        continue


                    if full:
                        filename = os.path.join("dataset", "S{}/Seq{}/imageSequence/full_video_{}/frame{:06}.jpg".format(self.S, self.Se, self.vid, k))
                    else:
                        filename = os.path.join("dataset", "S{}/Seq{}/imageSequence/video_{}/frame{:06}.jpg".format(self.S, self.Se, self.vid, k))

                    if end[0]-start[0]==end[1]-start[1]:
                        data[k] = {}
                        if save_img:
                            try:
                                cropped_frame = frame[start[1]:end[1], start[0]:end[0]]
                                if full:
                                    cv.imwrite(filename, frame)
                                else:
                                    cv.imwrite(filename, cropped_frame)
                            except FileExistsError:
                                pass

                        if save_txt:
                            data[k]["directory"] = filename
                            data[k]["2d_keypoints"] = self.imgPoint.reshape(1,-1)
                            data[k]["3d_keypoints"] = self.objPoint.reshape(1,-1)

            break
        if full:
            np.savez_compressed("dataset/S{}/Seq{}/imageSequence/full_video_{}".format(self.S,self.Se,self.vid), data)
        else:
            np.savez_compressed("dataset/S{}/Seq{}/imageSequence/video_{}".format(self.S,self.Se,self.vid), data)

        cap.release()

def save_frame(char):
    for seq in (1,2):
        for vid in [0,1,2,4,5,6,7,8]:
            v = All(char, seq, vid)
            v.save_cropped(True, True)

if __name__ == "__main__": 
    char_list = sys.argv[1]
    save_frame(char_list)