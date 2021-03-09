#!/usr/bin/python3

from matParser import *
from numpy.linalg import norm, matrix_rank
import numpy as np
import time, sys, math
import matplotlib.pyplot as plt

class Video(Joint):
    def __init__(self, S, Se, vid):
        super().__init__(S, Se, vid)
    
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

    def animate(self, show_skel=False, show_bbox=False, show_fps=False, show_axis=False):
        self.calib(0) # use first frame to set Camera Matrix for the entire video
        cap = cv.VideoCapture(self.avi_path)
        if (cap.isOpened()==False):
            print("Error opening the video file.")
        k = 0
        new_ft = 0
        prev_ft = 0
        nuc_list = []
        while (cap.isOpened()):
            if k == len(self.annot2D[self.camera][0]):
                print("Done!")
                break
            ret, frame = cap.read()
            if ret:
                self.get_joints(k)
                if show_skel:
                    for i in range(len(self.proj_xS)):
                        cv.circle(frame, (self.proj_xS[i],self.proj_yS[i]), radius=7, color=(255,255,255), thickness=-1)
                    for pair in self.joint_pairs:
                        pt1 = (self.proj_xS[pair[0]], self.proj_yS[pair[0]])
                        pt2 = (self.proj_xS[pair[1]], self.proj_yS[pair[1]])
                        cv.line(frame, pt1, pt2, color=(0,0,255), thickness=3)

                if show_bbox:
                    x1,y1,x2,y2 = self.draw_bbox(k)
                    start, end = (x1,y1), (x2,y2)
                    cv.rectangle(frame, start, end, color=(0,255,255), thickness=3)
                    print(end[0]-start[0], end[1]-start[1])

                self.get_arrow(k)
                if (self.valid_arrow(self.arrow_root) or self.valid_arrow(self.arrow_end)):
                    cv.arrowedLine(frame, self.arrow_root, self.arrow_end, (255,200,0), 10, 1, 0)

                if show_fps:
                    font = cv.FONT_HERSHEY_SIMPLEX
                    new_ft = time.time()
                    fps = 1/(new_ft-prev_ft)
                    prev_ft = new_ft
                    fps = str(round(fps,2))
                    cv.putText(frame, "FPS: {}".format(fps), (7,70), font, 3, (0,0,0), 10, cv.LINE_AA)
                    cv.putText(frame, "FPS: {}".format(fps), (7,70), font, 3, (0,10,200), 4, cv.LINE_AA)

                if show_axis:
                    self.draw_axis(frame)
                
                cv.imshow("Result on {}".format(self.avi_path), frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
            k += 1
            if k == 1000:
                break
        cap.release()
        cv.destroyAllWindows()

        x = [i for i in range(1, len(nuc_list)+1)]
        plt.plot(x, nuc_list)
        plt.xlabel("Frame")
        plt.ylabel("Sum of all bone length (mm)")
        plt.show()

if __name__ == "__main__":
    v = Video(1,1,1)
    v.animate(show_skel=True)

