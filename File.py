#!/usr/bin/python3

import scipy.io as sio
from numpy.linalg import norm
import numpy as np
import math
import cv2 as cv
import os
from tqdm import tqdm


class Video:
    def __init__(self, S, Se, vid):
        self.S = S
        self.Se = Se
        self.vid = vid

        self.mat_path = "dataset/S{}/Seq{}/annot.mat".format(S,Se)
        self.avi_path = "dataset/S{}/Seq{}/imageSequence/video_{}.avi".format(S,Se,vid)
        self.calib_path = "dataset/S{}/Seq{}/camera.calibration".format(S,Se)

        self.camera = vid # get camera number
        self.annot3D = sio.loadmat(self.mat_path)['annot3']
        self.annot2D = sio.loadmat(self.mat_path)['annot2']

    
    def __del__(self):
        print("Killed")
    
    def draw_bbox(self, nframe):
        coordinates = self.annot2D[self.camera][0][nframe]
        xS = []
        yS = []
        for k in range(0, len(coordinates)):
            if k%2 == 0:
                xS.append(coordinates[k])
            else:
                yS.append(coordinates[k])
        thresh = 100
        x1, y1 = int(min(xS)-thresh) , int(min(yS)-thresh)
        x2, y2 = int(max(xS)+thresh) , int(max(yS)+thresh) 

        w, h = x2-x1, y2-y1
        max_wh = np.max([w,h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        x1, y1 = x1-hp, y1-vp
        x2, y2 = x2+hp, y2+vp
        
        return x1,y1,x2,y2


    def get_cross(self, nframe):
        """
        Calculate the cross product given a frame

        Output: a 1x3 numpy array
        """
        objPoint = self.annot3D[self.camera][0][nframe]
        objPoint = np.array(objPoint.reshape(-1,3), dtype=np.float32)
        pt_list = [4, 9, 14]
        p0 = np.array(objPoint[pt_list[0]])
        p1 = np.array(objPoint[pt_list[1]])
        p2 = np.array(objPoint[pt_list[2]])
        n_vec = np.cross(p1-p0, p2-p0)
        self.n_vec = n_vec


    def cam_matrix(self):
        """
        Parse camera matrix from calibration file
        """
        calib = open(self.calib_path,"r")
        content = calib.readlines()
        content = [line.strip() for line in content]
        # 3x3 intrinsic matrix
        camMatrix = np.array(content[7*self.camera+5].split(" ")[3:], dtype=np.float32)
        camMatrix = np.reshape(camMatrix, (4,-1))
        camMatrix = camMatrix[0:3,0:3]
        self.camMatrix = camMatrix 
    

    def parse_frame(self, nframe):
        self.get_cross(nframe)
        self.objPoint = self.annot3D[self.camera][0][nframe]
        self.objPoint = np.array(self.objPoint.reshape(-1,3), dtype=np.float32)
        self.imgPoint = self.annot2D[self.camera][0][nframe]
        self.imgPoint = np.array(self.imgPoint.reshape(-1,2), dtype=np.float32)
        self.root = self.objPoint[4]


    def calib(self, nframe):
        self.cam_matrix()
        self.parse_frame(nframe)
        ret, rvec, tvec = cv.solvePnP(self.objPoint, self.imgPoint, self.camMatrix, np.zeros(4), flags=cv.SOLVEPNP_EPNP)
        
        assert ret
        self.rvec = rvec
        self.tvec = tvec
        self.dist = np.zeros(4)


    def get_arrow(self, nframe):
        self.parse_frame(nframe)
        arrow = []
        arrow.append(self.root)
        arrow.append(400*(self.n_vec/norm(self.n_vec))+self.root) # adjust normal vector length
        projected, _ = cv.projectPoints(np.float32(arrow), self.rvec, self.tvec, self.camMatrix, self.dist)
        arrow_root = (int(projected[0][0][0]), int(projected[0][0][1]))
        arrow_end = (int(projected[1][0][0]), int(projected[1][0][1]))
        self.arrow_root = arrow_root
        self.arrow_end = arrow_end
        self.angle = np.dot(self.n_vec, np.array([1,0,0]))/norm(self.n_vec)
        self.angle = math.acos(self.angle)*180/(math.pi)
    

    def valid_arrow(self, pt):
        """
        A boolean function that verifies whether a given point falls within the frame
        """
        return True if (0 <= pt[0] <= 2048) and (0 <= pt[1] <= 2048) else False


    def get_joints(self, nframe):
        self.parse_frame(nframe)
        projected, _ = cv.projectPoints(self.objPoint, self.rvec, self.tvec, self.camMatrix, self.dist)
        projected = projected.reshape(28,-1)
        proj_xS = []
        proj_yS = []
        for x,y in projected:
            proj_xS.append(int(x))
            proj_yS.append(int(y))
        self.proj_xS = proj_xS
        self.proj_yS = proj_yS


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
                                    cv.imwrite(filename, cv.resize(frame, (512,512), interpolation=cv.INTER_AREA))
                                else:
                                    cv.imwrite(filename, cropped_frame)
                            except FileExistsError:
                                pass

                        if save_txt:
                            data[k]["directory"] = filename
                            data[k]["bbox_start"] = start
                            data[k]["bbox_end"] = end
                            data[k]["pts_2d"] = self.imgPoint.reshape(-1,2)
                            data[k]["pts_3d"] = self.objPoint.reshape(-1,3)

            break
        if full:
            np.savez_compressed("dataset/S{}/Seq{}/imageSequence/full_video_{}".format(self.S,self.Se,self.vid), data)
        else:
            np.savez_compressed("dataset/S{}/Seq{}/imageSequence/video_{}".format(self.S,self.Se,self.vid), data)

        cap.release()


def save_frame(human):
    for seq in (1,2):
        for vid in [0,1,2,4,5,6,7,8]:
            v = All(human, seq, vid)
            v.save_cropped(False, True, False)

def merge_npz(human):
    merge_data = []
    for s in [1,2]:
        for k in [0,1,2,4,5,6,7,8]:
            npz = "./dataset/S{}/Seq{}/imageSequence/video_{}.npz".format(human,s,k)
            t = np.load(npz, allow_pickle=True)
            t = t['arr_0'].reshape(1,-1)
            merge_data.append(*t)
        filename = "./dataset/S{}/Seq1/imageSequence/S{}".format(human,human)
        np.savez_compressed(filename, merge_data)
    print("saved {}.npz".format(filename))


if __name__ == "__main__": 
    for human in range(1,9):
        save_frame(human)
        merge_npz(human)