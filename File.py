#!/usr/bin/python3

import scipy.io as sio
import numpy as np
import cv2 as cv
import os, sys
import numpy as np 
from tqdm import tqdm
from common.human import *


def get_rot_from_vecs(vec1: np.array, vec2: np.array) -> np.array:
    """ 
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector

    :return R: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    
    Such that vec2 = R @ vec1

    (Credit to Peter from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space)
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R
    

def convert_gt(gt_3d: np.array, t_info) -> np.array:
    """
    Compare GT3D kpts with T pose and obtain 16 rotation matrices

    :return R_stack: a (16,9) array with flattened rotation matrices for 16 bones
    """
    # process GT
    bone_info = vectorize(gt_3d)[:,:3] # (16,3) bone vecs

    num_bones = bone_info.shape[0]
    R_stack = np.zeros([num_bones, 9])
    # get rotation matrix for each bone
    for k in range(num_bones):
        R_stack[k,:] = get_rot_from_vecs(t_info[k,:], bone_info[k,:]).flatten()
    return R_stack



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


    def get_intrinsic(self):
        """
        Parse camera matrix from calibration file
        """
        calib = open(self.calib_path,"r")
        content = calib.readlines()
        content = [line.strip() for line in content]
        # 3x3 intrinsic matrix
        intrinsic = np.array(content[7*self.camera+5].split(" ")[3:], dtype=np.float32)
        intrinsic = np.reshape(intrinsic, (4,-1))
        self.intrinsic = intrinsic[:3, :3]


    def parse_frame(self, nframe):
        self.objPoint = self.annot3D[self.camera][0][nframe]
        self.objPoint = np.array(self.objPoint.reshape(-1,3), dtype=np.float32)
        self.imgPoint = self.annot2D[self.camera][0][nframe]
        self.imgPoint = np.array(self.imgPoint.reshape(-1,2), dtype=np.float32)
        self.root = self.objPoint[4]


    def calib(self, nframe):
        self.get_intrinsic()
        self.parse_frame(nframe)
        ret, rvec, tvec = cv.solvePnP(self.objPoint, self.imgPoint, self.intrinsic, np.zeros(4), flags=cv.SOLVEPNP_EPNP)
        
        assert ret
        self.rvec = rvec
        self.tvec = tvec
        self.dist = np.zeros(4)


    def get_joints(self, nframe):
        self.parse_frame(nframe)
        projected, _ = cv.projectPoints(self.objPoint, self.rvec, self.tvec, self.intrinsic, self.dist)
        projected = projected.reshape(28,-1)
        proj_xS = []
        proj_yS = []
        for x,y in projected:
            proj_xS.append(int(x))
            proj_yS.append(int(y))
        self.proj_xS = proj_xS
        self.proj_yS = proj_yS


    def to_camera_coordinate(self, pts_2d, pts_3d) -> np.array:
        self.get_intrinsic()
        ret, R, t= cv.solvePnP(pts_3d, pts_2d, self.intrinsic, np.zeros(4), flags=cv.SOLVEPNP_EPNP)

        # get extrinsic matrix
        assert ret
        R = cv.Rodrigues(R)[0]
        E = np.concatenate((R,t), axis=1)  # [R|t], a 3x4 matrix
    
        pts_3d = cv.convertPointsToHomogeneous(pts_3d).transpose().squeeze(1)
        cam_coor = E @ pts_3d
        cam_3d = cam_coor.transpose()
        return cam_3d


    def pop_joints(self, kpts):
        """
        Get 17 joints from the original 28 
        :param kpts: orginal kpts from MPI-INF-3DHP (an array of (28,n))
        :return new_skel: 
        """
        new_skel = np.zeros([17,3]) if kpts.shape[-1]==3 else np.zeros([17,2])
        ext_list = [2,4,5,6,         # spine+head
                    9,10,11,14,15,16,  # arms
                    18,19,20,23,24,25] # legs
        for row in range(1,17):
            new_skel[row, :] = kpts[ext_list[row-1], :]
        # interpolate clavicles to obtain vertebra
        new_skel[0, :] = (new_skel[5,:]+new_skel[8,:])/2
        return new_skel



class All(Video):
    def __init__(self, S, Se, vid):
        super().__init__(S, Se, vid)

    def __del__(self):
        print("Killed")

    def bound_number(self, x, y, frame_size):
    # make sure coordinates do not go beyond the frame
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
        """
        :param full: save original frame (in dimension of 2048x2048) if True
        """
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
                            h = Human(1.8, "cpu")
                            model = h.update_pose()
                            t_info = vectorize(model)[:,:3]
                            pts_2d, pts_3d = self.imgPoint.reshape(-1,2), self.objPoint.reshape(-1,3)
                            cam_3d = self.to_camera_coordinate(pts_2d, pts_3d)

                            data[k]["directory"] = filename
                            data[k]["bbox_start"] = start
                            data[k]["bbox_end"] = end
                            data[k]["pts_2d"] = self.pop_joints(pts_2d)
                            data[k]["pts_3d"] = self.pop_joints(pts_3d)
                            data[k]["cam_3d"] = self.pop_joints(cam_3d)
                            data[k]["vec_3d"] = convert_gt(cam_3d, t_info)
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