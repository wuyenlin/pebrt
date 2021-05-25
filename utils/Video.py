#!/usr/bin/python3

import scipy.io as sio
import numpy as np
import cv2 as cv

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
