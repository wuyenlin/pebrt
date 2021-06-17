import os
import numpy as np
import cv2 as cv
from tqdm import tqdm


class Video:
    def __init__(self, S, action, cam):
        self.S = S
        self.action = action
        self.cam = 54138969

        self.vid_path = "./h36m/S{}/Videos/{}.{}.mp4".format(S, action, cam)
        self.img_path = "./h36m/S{}/{}.{}/".format(S, action, cam)

        data_2d = np.load("./h36m/data_2d_h36m_gt.npz", allow_pickle=True)
        self.annot2D = data_2d["positions_2d"].reshape(1,-1)[0][0]["S"+str(S)][action][0]
        data_3d = np.load("./h36m/data_3d_h36m.npz", allow_pickle=True)
        self.annot3D = data_3d["positions_3d"].reshape(1,-1)[0][0]["S"+str(S)][action]

    
    def __del__(self):
        print("Killed")
    

    def draw_bbox(self, nframe):
        xS, yS = self.annot2D[nframe,:,0], self.annot2D[nframe,:,1]

        thresh = 100
        x1, y1 = int(min(xS)-thresh) , int(min(yS)-thresh)
        x2, y2 = int(max(xS)+thresh) , int(max(yS)+thresh) 

        w, h = x2-x1, y2-y1
        max_wh = np.max([w,h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        x1, y1 = x1-hp, y1-vp
        x2, y2 = x2+hp, y2+vp
        
        return x1, y1, x2, y2


    def bound_number(self, x, y, frame_size):
        """make sure coordinates do not go beyond the frame"""
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


    def save_cropped(self, save_img=False, save_npz=True):
        data = {}
        cap = cv.VideoCapture(self.vid_path)
        if (cap.isOpened() == False):
            print("Error opening the video file.")

        try:
            os.mkdir("./h36m/S{}/{}.{}"\
                        .format(self.S, self.action, self.cam))
        except FileExistsError:
            pass

        while (cap.isOpened()):
            print("processing S{}, {}...".format(self.S, self.action))
            for k in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
                ret, frame = cap.read()
                assert ret
                cv.imshow("Vid", frame)
                if cv.waitKey(25) & 0xFF == ord("q"):
                    break

                x1, y1, x2, y2 = self.draw_bbox(k)
                x1, y1 = self.bound_number(x1, y1, frame)
                x2, y2 = self.bound_number(x2, y2, frame)
                start = (x1, y1)
                end = (x2, y2)

                filename = "./h36m/S{}/{}.{}/frame{:06}.jpg"\
                            .format(self.S, self.action, self.cam, k)

                if end[0]-start[0] == end[1]-start[1]:
                    data[k] = {}
                    if save_img:
                        cropped_frame = frame[start[1]:end[1], start[0]:end[0]]
                        cv.imwrite(filename, cropped_frame)

                    if save_npz:
                        data[k]["directory"] = filename
                        data[k]["bbox_start"] = start
                        data[k]["bbox_end"] = end
                        data[k]["pts_2d"] = self.imgPoint.reshape(-1,2)
                        data[k]["pts_3d"] = self.objPoint.reshape(-1,3)
            break
        if save_npz:
            np.savez_compressed("./h36m/S{}".format(self.S), data)
        cap.release()


def merge_npz():
    merge_data = []
    files = ["./data_2d_h36m_gt.npz","./data_3d_h36m.npz"]
    print("Processing...")
    for item in files:
        t = np.load(item, allow_pickle=True)
        t = t["positions_2d"].reshape(1,-1) if item.endswith("2d_h36m_gt.npz") else t["positions_3d"].reshape(1,-1)
        merge_data.append(*t)
    filename = "./data_h36m"
    np.savez_compressed(filename, positions_2d=merge_data[0], positions_3d=merge_data[1])
    print("saved {}.npz".format(filename))


if __name__ == "__main__":
    # merge_npz()
    v = Video(1, "Directions", 54138969)
    v.save_cropped(True, False)