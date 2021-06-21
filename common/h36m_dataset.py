import os
import numpy as np
import cv2 as cv
from tqdm import tqdm



def merge_2d3d():
    """
    This function is for merging the 2D and 3D .npz file made by 
    data/prepare_data_h36m.py in VideoPose3D.
    Can be used for PEBRT.
    """
    merge_data = []
    files = ["./data_2d_h36m_gt.npz","./data_3d_h36m.npz"]
    print("Processing...")
    for item in files:
        t = np.load(item, allow_pickle=True)
        t = t["positions_2d"].reshape(1,-1) if item.endswith("2d_h36m_gt.npz") else t["positions_3d"].reshape(1,-1)
        merge_data.append(*t)
    filename = "./h36m/data_h36m"
    np.savez_compressed(filename, positions_2d=merge_data[0], positions_3d=merge_data[1])
    print("saved {}.npz".format(filename))


class Video:
    def __init__(self, S, action, cam):
        self.S = S
        self.action = action
        if S == "S1":
            action = action.replace("Photo", "TakingPhoto") \
                            .replace("WalkDog", "WalkingDog")
        self.vid_path = "./h36m/{}/Videos/{}.{}.mp4".format(S, action, cam)
        self.cam = 54138969

        self.img_path = "./h36m/{}/{}.{}/".format(S, self.action, cam)
        self.npz_name = "./h36m/{}/{}_{}".format(S, S, self.action)

        data_2d = np.load("./h36m/data_2d_h36m_gt.npz", allow_pickle=True)
        self.annot2D = data_2d["positions_2d"].reshape(1,-1)[0][0][S][self.action][0]
        data_3d = np.load("./h36m/data_3d_h36m.npz", allow_pickle=True)
        self.annot3D = data_3d["positions_3d"].reshape(1,-1)[0][0][S][self.action]

    
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


    def save(self, save_img=False, save_npz=True):
        data = {}
        cap = cv.VideoCapture(self.vid_path)
        if (cap.isOpened() == False):
            print("Error opening the video file: " + self.vid_path)

        try:
            os.mkdir("./h36m/{}/{}.{}"\
                        .format(self.S, self.action, self.cam))
        except FileExistsError:
            pass

        while (cap.isOpened()):
            print("processing {}, {}...".format(self.S, self.action))
            for k in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT))-1)):
                ret, frame = cap.read()
                assert ret
                if cv.waitKey(25) & 0xFF == ord("q"):
                    break

                try:
                    x1, y1, x2, y2 = self.draw_bbox(k)
                except IndexError:
                    continue
                x1, y1 = self.bound_number(x1, y1, frame)
                x2, y2 = self.bound_number(x2, y2, frame)
                start = (x1, y1)
                end = (x2, y2)

                filename = "./h36m/{}/{}.{}/frame{:06}.jpg"\
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
                        data[k]["positions_2d"] = self.annot2D.reshape(-1,2)
                        data[k]["positions_3d"] = self.annot3D.reshape(-1,3)
            cap.release()
        return data if save_npz else None


def main(subject_action):
    """
    This function is for merging the 2D and 3D .npz file made by 
    data/prepare_data_h36m.py in VideoPose3D.
    Can be used for PEBRT.
    """
    tabs = {}
    for s in subject_action.keys():
        for action in subject_action[s]:
            v = Video(s, action, 54138969)
            data = (v.save(save_img=False, save_npz=True))
            cat = s + "/" + action
            tabs[cat] = data
    print("Merging all npz files.")
    filename = "./h36m/data_h36m_frame"
    np.savez_compressed(filename, **tabs)
    print("saved {}.npz".format(filename))
    print("Done!")


if __name__ == "__main__":
    subject_action = {
        "S1": ["Photo", "Phoning", "Sitting 1", "Purchases", "Purchases 1", "WalkTogether", "Sitting 2", "WalkDog", 
                "Smoking 1", "Phoning 1", "Walking 1", "Walking", "Discussion 1", "SittingDown", "Directions", 
                "Greeting 1", "Eating 2", "Eating", "Photo 1", "WalkTogether 1", 
                "Greeting", "Directions 1", "WalkDog 1", "Posing 1", "Waiting", "Posing", 
                "Discussion", "Smoking", "Waiting 1", "SittingDown 2"],

        "S5": ["Phoning", "Sitting 1", "SittingDown 1", "Purchases", "Purchases 1", 
                "WalkTogether", "Discussion 3", "Sitting", "Waiting 2", "Smoking 1", 
                "Photo 2", "Phoning 1", "Walking 1", "Eating 1", "WalkDog 1", "Walking", 
                "SittingDown", "Greeting 1", "Eating", "WalkTogether 1", "Greeting 2", 
                "Directions 2", "Directions 1", "Posing 1", "Discussion 2", 
                "Photo", "WalkDog", "Posing", "Smoking", "Waiting 1"],

        "S6": ["Phoning", "Sitting 1", "SittingDown 1", "Purchases", "Purchases 1", 
                "WalkTogether", "Sitting 2", "Smoking 1", "Waiting 3", "Phoning 1", 
                "Photo 1", "Walking 1", "Eating 1", "WalkDog 1", "Walking", "Discussion 1", 
                "SittingDown", "Directions", "Greeting 1", "Eating 2", "WalkTogether 1", "Greeting", 
                "Directions 1", "Posing 2", "Waiting", "Photo", "WalkDog", "Posing", "Discussion", "Smoking"],

        "S7": ["Phoning", "Phoning 2", "Sitting 1", "SittingDown 1", "Purchases", "Purchases 1", "WalkTogether", 
                "Sitting", "Waiting 2", "Smoking 1", "Photo 1", "Walking 1", "Eating 1", "WalkDog 1", 
                "Discussion 1", "SittingDown", "Directions", "Greeting 1", "Eating", "WalkTogether 1", 
                "Greeting", "Directions 1", "Posing 1", "Photo", "WalkDog", "Posing", 
                "Discussion", "Smoking", "Waiting 1", "Walking 2"],

        "S8": ["Phoning", "Sitting 1", "SittingDown 1", "Purchases", "Purchases 1", "Sitting", "Smoking 1", 
                "Phoning 1", "Photo 1", "Walking 1", "Eating 1", "WalkDog 1", "Walking", "Discussion 1", "SittingDown", 
                "Directions", "Greeting 1", "WalkTogether 2", "Eating", "WalkTogether 1", "Greeting", "Directions 1", 
                "Posing 1", "Waiting", "Photo", "WalkDog", "Posing", "Discussion", "Smoking", "Waiting 1"],

        "S9": ["Phoning", "Sitting 1", "SittingDown 1", "Purchases", "Purchases 1", "WalkTogether", "Sitting", "Smoking 1", 
                "Phoning 1", "Photo 1", "Walking 1", "Eating 1", "WalkDog 1", "Walking", "Discussion 1", "SittingDown", 
                "Directions", "Greeting 1", "Eating", "WalkTogether 1", "Greeting", "Directions 1", "Posing 1", "Discussion 2", 
                "Waiting", "Photo", "WalkDog", "Posing", "Smoking", "Waiting 1"],

        "S11": ["Phoning 2", "Sitting 1", "SittingDown 1", "Purchases", "Purchases 1", "WalkTogether", "Sitting", "Photo 1", 
                "Walking 1", "Eating 1", "WalkDog 1", "Walking", "Discussion 1", "Phoning 3", "Smoking 2", "SittingDown", 
                "Eating", "WalkTogether 1", "Greeting", "Greeting 2", "Posing 1", "Discussion 2", "Waiting", 
                "Photo", "WalkDog", "Posing", "Smoking", "Waiting 1"],
    }
    main(subject_action)