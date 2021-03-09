#!/usr/bin/python3

import os, json, csv, sys
from numpy.linalg import norm
import numpy as np 
from tqdm import tqdm
from File import *

class All(File):
    def __init__(self, S, Se, vid):
        super().__init__(S, Se, vid)

    def __del__(self):
        print("Killed")

    def save_cropped(self, save_img=False, save_txt=True):
        data = {}
        cap = cv.VideoCapture(self.avi_path)
        if (cap.isOpened()==False):
            print("Error opening the video file.")
        try:
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

                    filename = os.path.join("dataset", "S{}/Seq{}/imageSequence/video_{}/frame{:06}.jpg".format(self.S, self.Se, self.vid, k))
                    if end[0]-start[0]==end[1]-start[1]:
                        data[k] = {}
                        if save_img:
                            try:
                                cropped_frame = frame[start[1]:end[1], start[0]:end[0]]
                                cv.imwrite(filename, cropped_frame)
                            except FileExistsError:
                                pass

                        if save_txt:
                            data[k]["directory"] = filename
                            data[k]["keypoints"] = self.objPoint.reshape(1,-1)
            break
        np.savez_compressed("dataset/S{}/Seq{}/imageSequence/video_{}".format(self.S,self.Se,self.vid), data)
        cap.release()

# char_list = [sys.argv[1], sys.argv[2]]
def save_frame(char_list):
    for char in char_list:
        for seq in (1,2):
            for vid in [0,1,2,4,5,6,7,8]:
                v = All(char, seq, vid)
                v.save_cropped(True, True)

def merge_npz():
    merge_data = []
    for k in [0,1,2,4,5,6,7,8]:
        npz = "dataset/S1/Seq1/imageSequence/video_{}.npz".format(k)
        t = np.load(npz, allow_pickle=True)
        t = t['arr_0'].reshape(1,-1)
        merge_data.append(*t)
    np.savez_compressed("dataset/S1/Seq1/imageSequence/S1seq1", merge_data)
    print("saved")


if __name__ == "__main__": 
    merge_npz()